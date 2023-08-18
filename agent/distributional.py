from absl import flags
from typing import Optional, Sequence, Union
import enum
import math

from acme.tf.networks import distributions as ad
from acme.tf.networks import DiscreteValuedHead, CriticMultiplexer, LayerNormMLP
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import dtype_util
import numpy as np
from scipy.stats import norm

tfd = tfp.distributions


uniform_initializer = tf.initializers.VarianceScaling(
    distribution='uniform', mode='fan_out', scale=0.333)


FLAGS = flags.FLAGS


class RiskDiscreteValuedDistribution(ad.DiscreteValuedDistribution):
    def __init__(self,
                 values: tf.Tensor,
                 logits: Optional[tf.Tensor] = None,
                 probs: Optional[tf.Tensor] = None,
                 name: str = 'RiskDiscreteValuedDistribution'):
        super().__init__(values, logits, probs, name)

    def _normal_dist_volc(self, quantile):
        prob_density = round(norm.ppf(quantile), 4)
        return prob_density

    def meanstd(self) -> tf.Tensor:
        """Implements mean-volc*std"""
        volc = FLAGS.std_coef
        return self.mean() - volc*self.stddev()

    def var(self, th) -> tf.Tensor:
        """Implements mean-volc*std for VaR estimation"""
        volc = self._normal_dist_volc(th)
        return self.mean() - volc*self.stddev()

    def cvar(self, th) -> tf.Tensor:
        quantile = 1 - th
        cdf = tf.cumsum(self.probs_parameter(), axis=-1)
        exclude_logits = cdf > quantile
        zero = np.array(0, dtype=dtype_util.as_numpy_dtype(cdf.dtype))
        clogits = tf.where(exclude_logits, zero, self.probs_parameter())
        return tf.reduce_sum(clogits * self.values, axis=-1)


class RiskDiscreteValuedHead(DiscreteValuedHead):
    def __init__(self,
                 vmin: Union[float, np.ndarray, tf.Tensor],
                 vmax: Union[float, np.ndarray, tf.Tensor],
                 num_atoms: int,
                 w_init: Optional[snt.initializers.Initializer] = None,
                 b_init: Optional[snt.initializers.Initializer] = None):
        super().__init__(vmin, vmax, num_atoms, w_init, b_init)

    def __call__(self, inputs: tf.Tensor) -> RiskDiscreteValuedDistribution:
        logits = self._distributional_layer(inputs)
        logits = tf.reshape(logits,
                            tf.concat([tf.shape(logits)[:1],  # batch size
                                       tf.shape(self._values)],
                                      axis=0))
        values = tf.cast(self._values, logits.dtype)

        return RiskDiscreteValuedDistribution(values=values, logits=logits)


def quantile_project(  # pylint: disable=invalid-name
    q: tf.Tensor,
    v: tf.Tensor,
    q_grid: tf.Tensor,
) -> tf.Tensor:
    """Project quantile distribution (quantile_grid, values) onto quantile under the L2-metric over CDFs.

    This projection works for any support q.
    Let Kq be len(q_grid)

    Args:
    q: () quantile
    v: (batch_size, Kq) values to project onto
    q_grid:  (Kq,) Quantiles for P(Zp[i])

    Returns:
    Quantile projection of (q_grid, v) onto q.
    """

    # Asserts that Zq has no leading dimension of size 1.
    if q_grid.get_shape().ndims > 1:
        q_grid = tf.squeeze(q_grid, axis=0)
    q = q[None]
    # Extracts vmin and vmax and construct helper tensors from Zq.
    vmin, vmax = q_grid[0], q_grid[-1]
    d_pos = tf.concat([q_grid, vmin[None]], 0)[1:]
    d_neg = tf.concat([vmax[None], q_grid], 0)[:-1]

    # Clips Zp to be in new support range (vmin, vmax).
    clipped_q = tf.clip_by_value(q, vmin, vmax)  # (1,)
    eq_mask = tf.cast(tf.equal(q_grid, q), q_grid.dtype)
    if tf.equal(tf.reduce_sum(eq_mask), 1.0):
        # (batch_size, )
        return tf.squeeze(tf.boolean_mask(v, eq_mask, axis=1), axis=-1)

    # need interpolation
    pos_neg_mask = tf.cast(tf.roll(q_grid <= q, 1, axis=0), q_grid.dtype) \
        * tf.cast(tf.roll(q_grid >= q, -1, axis=0), q_grid.dtype)
    pos_neg_v = tf.boolean_mask(v, pos_neg_mask, axis=1)    # (batch_size, 2)

    # Gets the distance between atom values in support.
    d_pos = (d_pos - q_grid)[None, :]  # (1, Kq)
    d_neg = (q_grid - d_neg)[None, :]  # (1, Kq)

    clipped_q_grid = q_grid[None, :]  # (1, Kq)
    delta_qp = clipped_q - clipped_q_grid  # (1, Kq)

    d_sign = tf.cast(delta_qp >= 0., dtype=v.dtype)
    delta_hat = (d_sign * delta_qp / d_pos) - \
        ((1. - d_sign) * delta_qp / d_neg)  # (1, Kq)
    # (batch_size, )
    return tf.reduce_sum(tf.clip_by_value(1. - delta_hat, 0., 1.) * v, 1)


@tfp.experimental.register_composite
class QuantileDistribution(tfd.Categorical):
    def __init__(self,
                 values: tf.Tensor,
                 quantiles: tf.Tensor,
                 probs: tf.Tensor,
                 name: str = 'QuantileDistribution'):
        """Quantile Distribution
        values: (batch_size, Kq)
        quantiles: (Kq,) or (batch_size, Kq)
        probs: (Kq,)
        """
        self._quantiles = tf.convert_to_tensor(quantiles)
        self._shape_strings = [f'D{i}' for i, _ in enumerate(quantiles.shape)]
        self._values = tf.convert_to_tensor(values)
        self._probs = tf.convert_to_tensor(probs)

        super().__init__(probs=probs, name=name)
        self._parameters = dict(values=values,
                                quantiles=quantiles,
                                probs=probs,
                                name=name)

    @property
    def quantiles(self) -> tf.Tensor:
        return self._quantiles

    @property
    def values(self) -> tf.Tensor:
        return self._values

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return dict(
            values=tfp.util.ParameterProperties(
                event_ndims=lambda self: self.quantiles.shape.rank),
            quantiles=tfp.util.ParameterProperties(
                event_ndims=None),
            probs=tfp.util.ParameterProperties(
                event_ndims=None))

    def _sample_n(self, n, seed=None) -> tf.Tensor:
        indices = super()._sample_n(n, seed=seed)
        return tf.gather(self.values, indices, axis=-1)

    def _mean(self) -> tf.Tensor:
        # assume values are always with equal prob
        return tf.reduce_mean(self.values, axis=-1)

    def _variance(self) -> tf.Tensor:
        dist_squared = tf.square(tf.expand_dims(self.mean(), -1) - self.values)
        return tf.reduce_sum(self.probs_parameter() * dist_squared, axis=-1)

    def _event_shape(self):
        # Omit the atoms axis, to return just the shape of a single (i.e. unbatched)
        # sample value.
        return self._quantiles.shape[:-1]

    def _event_shape_tensor(self):
        return tf.shape(self._quantiles)[:-1]

    def meanstd(self) -> tf.Tensor:
        """Implements mean-volc*std"""
        volc = FLAGS.std_coef
        return self.mean() - volc*self.stddev()

    def var(self, th) -> tf.Tensor:
        quantile = tf.convert_to_tensor(1 - th)
        return quantile_project(quantile, self._values, self.quantiles)

    def cvar(self, th) -> tf.Tensor:
        quantile = 1 - th
        cdf = tf.cumsum(self.probs_parameter(), axis=-1)
        exclude_probs = cdf > quantile
        zero = np.array(0, dtype=dtype_util.as_numpy_dtype(cdf.dtype))
        cprobs = tf.where(exclude_probs, zero, self.probs_parameter())
        return tf.reduce_sum(cprobs * self.values, axis=-1)

    def transform(self, shift: tf.Tensor, scale: tf.Tensor) -> tf.Tensor:
        return QuantileDistribution(
            shift + scale*self._values,
            self._quantiles,
            self._probs,
            name='Shifted_QuantileDistribution')


class QuantileDistProbType(enum.Enum):
    LEFT = 1
    MID = 2
    RIGHT = 3


class QuantileDiscreteValuedHead(snt.Module):
    def __init__(self,
                 quantiles: np.ndarray,
                 prob_type: QuantileDistProbType = QuantileDistProbType.MID,
                 w_init: Optional[snt.initializers.Initializer] = None,
                 b_init: Optional[snt.initializers.Initializer] = None):
        super().__init__(name='QuantileDiscreteValuedHead')
        self._quantiles = tf.convert_to_tensor(quantiles)
        assert quantiles[0] > 0
        assert quantiles[-1] < 1.0
        left_probs = quantiles - np.insert(quantiles[:-1], 0, 0.0)
        right_probs = np.insert(
            quantiles[1:], len(quantiles)-1, 1.0) - quantiles
        if prob_type == QuantileDistProbType.LEFT:
            probs = left_probs
        elif prob_type == QuantileDistProbType.MID:
            probs = (left_probs + right_probs) / 2
        elif prob_type == QuantileDistProbType.RIGHT:
            probs = right_probs
        self._probs = tf.convert_to_tensor(probs)
        self._distributional_layer = snt.Linear(tf.size(self._quantiles),
                                                w_init=w_init,
                                                b_init=b_init)

    def __call__(self, inputs: tf.Tensor) -> tfd.Distribution:
        quantile_values = self._distributional_layer(inputs)
        quantile_values = tf.reshape(quantile_values,
                                     tf.concat([tf.shape(quantile_values)[:1],
                                                tf.shape(self._quantiles)],
                                               axis=0))
        quantiles = tf.cast(self._quantiles, quantile_values.dtype)
        probs = tf.cast(self._probs, quantile_values.dtype)
        return QuantileDistribution(values=quantile_values, quantiles=quantiles,
                                    probs=probs)


class IQNHead(snt.Module):
    def __init__(self, th, n_cos=64, n_tau=8, n_k=32, layer_size: int = 256,
                 quantiles: np.ndarray = np.arange(0.01, 1.0, 0.01),
                 prob_type: QuantileDistProbType = QuantileDistProbType.MID,
                 w_init: Optional[snt.initializers.Initializer] = uniform_initializer):
        super().__init__(name='IQNHead')
        self._th = th
        self._n_cos = n_cos
        self._n_tau = n_tau
        self._n_k = n_k
        self._pis = tf.convert_to_tensor(
            [np.pi*i for i in range(self._n_cos)])[None, None, :]
        self._phi = snt.nets.MLP(
            (layer_size,),
            w_init=w_init,
            activation=tf.nn.relu,
            activate_final=True)
        self._f = snt.nets.MLP(
            (layer_size, 1),
            w_init=w_init,
            activation=tf.nn.elu,
            activate_final=False
        )
        self._quantiles = tf.convert_to_tensor(quantiles)
        assert quantiles[0] > 0
        assert quantiles[-1] < 1.0
        left_probs = quantiles - np.insert(quantiles[:-1], 0, 0.0)
        right_probs = np.insert(
            quantiles[1:], len(quantiles)-1, 1.0) - quantiles
        if prob_type == QuantileDistProbType.LEFT:
            probs = left_probs
        elif prob_type == QuantileDistProbType.MID:
            probs = (left_probs + right_probs) / 2
        elif prob_type == QuantileDistProbType.RIGHT:
            probs = right_probs
        self._probs = tf.convert_to_tensor(probs)

    def __call__(self, inputs: tf.Tensor, policy=False) -> tfd.Distribution:
        if not policy:
            # (batch, n_tau, 1)
            taus = tf.random.uniform(
                (inputs.shape[0], self._n_tau, 1), 0, 1, dtype=inputs.dtype)
        else:
            # (batch, n_k, 1)
            taus = tf.random.uniform(
                (inputs.shape[0], self._n_k, 1), 0, 1, dtype=inputs.dtype)*(1-self._th)
        cos = tf.cos(taus*self._pis)        # (batch, n_tau, n_cos)
        cos_x = self._phi(cos)              # (batch, n_tau, n_layer)
        x = tf.expand_dims(inputs, axis=1)  # (batch, 1, n_layer)
        icdf = self._f(x*cos_x)             # (batch, n_tau, 1)
        if not policy:
            taus = tf.transpose(taus, [2, 0, 1])  # (1, batch, n_tau)
        probs = tf.cast(self._probs, inputs.dtype)
        return QuantileDistribution(values=tf.squeeze(icdf, axis=-1), quantiles=taus, probs=probs)


class IQNCritic(snt.Module):
    def __init__(self, th, n_cos=64, n_tau=8, n_k=32,
                 critic_layer_sizes: Sequence[int] = (512, 512, 256),
                 quantiles: np.ndarray = np.arange(0.01, 1.0, 0.01),
                 prob_type: QuantileDistProbType = QuantileDistProbType.MID,
                 w_init: Optional[snt.initializers.Initializer] = uniform_initializer):
        super().__init__(name='IQNCritic')
        self._head = snt.Sequential([
            # The multiplexer concatenates the observations/actions.
            CriticMultiplexer(),
            LayerNormMLP(critic_layer_sizes, activate_final=True)])
        self._iqn = IQNHead(th, n_cos, n_tau, n_k,
                            critic_layer_sizes[-1], quantiles, prob_type, w_init)

    def __call__(self, observation: tf.Tensor, action: tf.Tensor, policy=False) -> tfd.Distribution:
        return self._iqn(self._head(observation, action), policy)


class QuantileLoss(snt.Module):
    def __init__(self, loss_type='huber', b_decay=0.9, name: Optional[str] = None):
        super().__init__(name=name)
        self.loss_type = loss_type
        self.b_decay = b_decay
        self.b = tf.Variable(1.0, dtype=tf.float32, name='b', trainable=False)

    def huber(self, x: tf.Tensor, k=1.0):
        return tf.where(tf.abs(x) < k, 0.5 * tf.pow(x, 2), k * (tf.abs(x) - 0.5 * k))


    def gaussian_loss(self, td_error: tf.Tensor, b: tf.Tensor):
        abs_u = tf.abs(td_error)
        def f(x): return (1.0 + tf.math.erf(x / tf.sqrt(2.0))) / 2.0
        phi = f(-abs_u/b)
        loss = tf.multiply(abs_u, (1.0 - 2*phi)) + b*tf.sqrt(2.0/math.pi) * \
            tf.exp(-tf.pow(abs_u, 2.0)/(2*b*b)) - b*math.sqrt(2.0/math.pi)
        return loss


    def gaussian_loss_taylor(self, td_error: tf.Tensor, b: tf.Tensor):
        abs_u = tf.abs(td_error)
        loss = tf.where(abs_u <= b, tf.pow(abs_u, 2.0) /
                        (b*math.sqrt(2.0*math.pi)), abs_u - b*math.sqrt(2.0/math.pi))
        return loss


    def laplace_loss(self, td_error: tf.Tensor, b: tf.Tensor):
        abs_u = tf.abs(td_error)
        loss = abs_u + b*tf.exp(-abs_u/b)-b
        return loss


    def laplace_loss_taylor(self, td_error: tf.Tensor, b: tf.Tensor):
        abs_u = tf.abs(td_error)
        loss = tf.where(abs_u <= b, tf.pow(abs_u, 2.0) / (2*b), abs_u - b)
        return loss

    def __call__(self, 
                q_tm1: QuantileDistribution, 
                r_t: tf.Tensor,
                d_t: tf.Tensor,
                q_t: QuantileDistribution):
        """Implements Quantile Regression Loss
        q_tm1: critic distribution of t-1
        r_t:   reward
        d_t:   discount
        q_t:   target critic distribution of t
        loss_type: 'huber', 'gl', 'gl-tl', 'lapl', 'lapl-tl'
        """

        z_t = tf.reshape(r_t, (-1, 1)) + tf.reshape(d_t, (-1, 1)) * q_t.values
        z_tm1 = q_tm1.values
        diff = tf.expand_dims(tf.transpose(z_t), -1) - \
            z_tm1    # (n_tau_p, n_batch, n_tau)
        diff_detach = tf.stop_gradient(diff)

        if self.loss_type == 'huber':
            k = 1
            loss = self.huber(diff, k) / k
        else:
            std1 = tf.math.reduce_std(
                tf.cast(z_t, dtype=tf.float32), 1)      # (n_batch,1)
            std2 = tf.math.reduce_std(
                tf.cast(z_tm1, dtype=tf.float32), 1)   # (n_batch,1)
            self.b.assign(self.b*self.b_decay + (1 - self.b_decay)* tf.reduce_mean(tf.abs(std1 - std2)))
            b = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)
            b = b.write(0, self.b.value())
            b = tf.cast(b.stack(), tf.float32)
            if self.loss_type == 'gl':
                loss = self.gaussian_loss(diff, b)
            elif self.loss_type == 'gl_tl':
                loss = self.gaussian_loss_taylor(diff, b)
            elif self.loss_type == 'lapl':
                loss = self.laplace_loss(diff, b)
            elif self.loss_type == 'lapl_tl':
                loss = self.laplace_loss_taylor(diff, b)

        loss *= tf.abs(q_tm1.quantiles -
                    tf.cast(diff_detach < 0, diff_detach.dtype))  # quantile regression loss
        return tf.reduce_mean(loss, (0, -1))
