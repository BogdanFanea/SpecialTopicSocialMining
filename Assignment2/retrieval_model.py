"""NumPy implementation of the FOP reference blocks and meme head."""
import numpy as np


def normalize(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    unit = x / np.maximum(norm, 1e-12)

    def backward(grad):
        radial = (grad * unit).sum(axis=1, keepdims=True) * unit
        return (grad - np.where(norm > 1e-12, radial, 0)) / np.maximum(norm, 1e-12)

    return unit, backward


def orthogonal_loss(z, labels):
    q, backward = normalize(z)
    similarity = q @ q.T
    same = labels[:, None] == labels[None, :]
    positive = same & ~np.eye(len(labels), dtype=bool)
    negative = ~same
    pos_count = max(1, positive.sum())
    neg_count = max(1, negative.sum())
    value = 1 - similarity[positive].sum() / pos_count
    value += 0.7 * np.abs(similarity[negative]).sum() / neg_count
    pair_grad = np.zeros_like(similarity)
    pair_grad[positive] = -1 / pos_count
    pair_grad[negative] = (0.7 / neg_count) * np.sign(similarity[negative])
    return value, backward((pair_grad + pair_grad.T) @ q)


class FOP:
    def __init__(self, dims, d, gate_hidden, dropout, fusion, seed):
        self.params, self.stats = {}, {}
        self.rng = np.random.default_rng(seed)
        self.dropout = dropout
        self.fusion = fusion
        self.add_block("image", dims[0], d)
        self.add_block("text", dims[1], d)
        self.add_linear("head", d, 2)
        if fusion == "gated":
            self.add_block("attention", 2 * d, gate_hidden)
            self.add_linear("gate", gate_hidden, d)
        else:
            self.params["weights"] = self.rng.random(2, dtype=np.float32)

    def add_linear(self, name, inputs, outputs):
        bound = np.sqrt(6 / (inputs + outputs))
        self.params[name + "_w"] = self.rng.uniform(-bound, bound, (inputs, outputs)).astype("f")
        self.params[name + "_b"] = np.full(outputs, 0.01, dtype="f")

    def add_block(self, name, inputs, outputs):
        self.add_linear(name, inputs, outputs)
        self.params[name + "_gamma"] = np.ones(outputs, dtype="f")
        self.params[name + "_beta"] = np.zeros(outputs, dtype="f")
        self.stats[name + "_mean"] = np.zeros(outputs, dtype="f")
        self.stats[name + "_variance"] = np.ones(outputs, dtype="f")

    def linear(self, name, x):
        w, b = name + "_w", name + "_b"

        def backward(grad):
            self.grads[w] = x.T @ grad
            self.grads[b] = grad.sum(axis=0)
            return grad @ self.params[w].T

        return x @ self.params[w] + self.params[b], backward

    def block(self, name, x, training, dropout):
        a, linear_back = self.linear(name, x)
        mean_key, var_key = name + "_mean", name + "_variance"
        gamma, beta = name + "_gamma", name + "_beta"
        if training:
            mean, variance = a.mean(0), a.var(0)
            self.stats[mean_key] *= 0.9
            self.stats[mean_key] += 0.1 * mean
            self.stats[var_key] *= 0.9
            self.stats[var_key] += (0.1 * len(a) / (len(a) - 1)) * variance
        else:
            mean, variance = self.stats[mean_key], self.stats[var_key]
        inv_std = 1 / np.sqrt(variance + 1e-5)
        normalized = (a - mean) * inv_std
        pre_relu = normalized * self.params[gamma] + self.params[beta]
        mask = pre_relu > 0
        scale = 1
        if training and dropout:
            scale = (self.rng.random(pre_relu.shape, dtype=np.float32) >= dropout) / np.float32(1 - dropout)

        def backward(grad):
            grad = grad * scale * mask
            self.grads[gamma] = (grad * normalized).sum(0)
            self.grads[beta] = grad.sum(0)
            grad = grad * self.params[gamma]
            return linear_back(inv_std * (grad - grad.mean(0) - normalized * (grad * normalized).mean(0)))

        return np.maximum(pre_relu, 0) * scale, backward

    def forward(self, image, text, training=False):
        u, image_back = self.block("image", image, training, self.dropout)
        v, text_back = self.block("text", text, training, self.dropout)
        u, u_back = normalize(u)
        v, v_back = normalize(v)
        if self.fusion == "gated":
            hidden, hidden_back = self.block("attention", np.concatenate((u, v), axis=1), training, 0)
            raw_gate, gate_back = self.linear("gate", hidden)
            gate = 1 / (1 + np.exp(-raw_gate))
            a, b = np.tanh(u), np.tanh(v)
            z = gate * a + (1 - gate) * b
        else:
            w = self.params["weights"]
            z = w[0] * u + w[1] * v
        logits, head_back = self.linear("head", z)

        def backward(logit_grad, extra_grad):
            self.grads = {}
            grad = head_back(logit_grad) + extra_grad
            if self.fusion == "gated":
                gate_grad = grad * (a - b) * gate * (1 - gate)
                du, dv = np.split(hidden_back(gate_back(gate_grad)), 2, axis=1)
                du = du + grad * gate * (1 - a * a)
                dv = dv + grad * (1 - gate) * (1 - b * b)
            else:
                self.grads["weights"] = np.array([(grad * u).sum(), (grad * v).sum()])
                du, dv = grad * w[0], grad * w[1]
            image_back(u_back(du))
            text_back(v_back(dv))
            return self.grads

        return logits, z, backward


def cross_entropy(logits, labels=None):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    partition = exp.sum(axis=1, keepdims=True)
    scores = exp / partition
    if labels is None:
        return scores[:, 1]
    rows = np.arange(len(labels))
    loss = (np.log(partition[:, 0]) - shifted[rows, labels]).mean()
    scores[rows, labels] -= 1
    return loss, scores / len(labels)
