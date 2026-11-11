"""
This module implements custom PyTorch Autograd functions which can be used to facilitate weight updating for
Semi-Positive Definite Matrices on a Stiefel Manifold via Riemannian Geometry, as opposed to Euclidean Space.
The functions are then used to define an SPDNet layer.

This is described fully in:

Huang, Z., & Van Gool, L. (2016). A Riemannian Network for SPD Matrix Learning, 2036–2042.
https://doi.org/10.1109/CVPR.2014.132

Ionescu, C.; Vantzos, O.; and Sminchisescu, C. 2015. Matrix back- propagation for deep networks with structured layers.
In ICCV. Jarrett,
https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Ionescu_Matrix_Backpropagation_for_ICCV_2015_paper.pdf
"""
import torch
nn = torch.nn


def sym_op(X):
    """Symmetrizes a matrix."""
    return 0.5 * (X + X.t())


def diag_op(X):
    """Zeroes out the non-diagonal elements of a matrix."""
    X[torch.eye(X.shape[0]) == 0] = 0
    return X


class BiMap(torch.autograd.Function):
    """
    Defines Bilinear Layer whose weights are updated with an orthogonality constraint. Can be used to generate
    CNN-like SPD filters which convert an input SPD matrix into another learned SPD matrix
    """
    @staticmethod
    def forward(ctx, X, W):
        """
        Performs forward pass of Bilinear transformation of an SPD matrix with dimension reduction Weights
        :param ctx: context object
        :param X: Tensor, input SPD matrix (d x d)
        :param W: Tensor, Orthogonal Weights (orthogonality is necessary for Steifel manifold gradient search) (m x d, m < d)
        :return: Tensor
        """
        

        out = torch.mm(torch.mm(W, X), W.t())
        ctx.save_for_backward(X, W)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes input gradients for BiMap Bilinear transformation.  The gradient for the transformations weights
        represent the Riemannian gradient with respect to the tangent of the Stiefel manifold
        NOTE: Results are only valid if input (X) to forward is a symmetric matrix!
        """
        X, W = ctx.saved_tensors
        grad_X = torch.mm(torch.mm(W.t(), grad_output), W)
        grad_W = 2 * grad_output.mm(W.mm(X))
        grad_W_r = grad_W - grad_W.mm(W.t().mm(W))
        return grad_X, grad_W_r


class EigDecomp(torch.autograd.Function):
    """
    实现 ReEig 非线性操作，通过对 SPD 矩阵的特征值设置下限来进行正则化。
    """

    @staticmethod
    def forward(ctx, X):
        # 使用 torch.linalg.eigh 替代 torch.linalg.eig
        S, U = torch.linalg.eigh(X)  # 返回实数特征值和特征向量
        S = S.diag()  # 将特征值转换为对角矩阵
        ctx.save_for_backward(U, S)  # 保存特征值和特征向量以供反向传播使用
        return S, U

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        反向传播：计算输入矩阵的梯度。
        :param ctx: 上下文对象，包含前向传播中保存的变量
        :param grad_outputs: 从下一层传递回来的梯度
        :return: 输入矩阵的梯度
        """
        grad_S, grad_U = grad_outputs  # 获取梯度
        U, S = ctx.saved_tensors  # 恢复前向传播中保存的特征值和特征向量
        rank = S.shape[0]  # 矩阵的秩

        # 计算特征值梯度的辅助矩阵 P
        s_vals = S.diag()  # 提取特征值
        P = 1 / (s_vals.view(-1, 1).repeat((1, rank)) - s_vals.repeat(rank, 1))  # 计算特征值差的倒数
        P[torch.eye(rank) == 1] = 0  # 对角线元素置为 0，避免除以 0

        # 计算特征向量的梯度 dU 和特征值的梯度 dS
        dU = 2 * U.mm((P.t() * sym_op(U.t().mm(grad_U))).mm(U.t()))
        dS = U.mm(diag_op(grad_S).mm(U.t()))

        return dU + dS


class EigenLog(torch.autograd.Function):
    """
    实现 LogEig 非线性操作，为输入的 SPD 矩阵引入对数欧几里得度量。
    """
    @staticmethod
    def forward(ctx, S, U):
        S_log = torch.log(S.diag()).diag()  # 对特征值取对数
        ctx.save_for_backward(S, S_log, U)  # 保存特征值、对数特征值和特征向量以供反向传播使用
        return U.mm(S_log.mm(U.t()))  # 重构对数欧几里得 SPD 矩阵

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        反向传播：计算输入矩阵的梯度。
        :param ctx: 上下文对象，包含前向传播中保存的变量
        :param grad_outputs: 从下一层传递回来的梯度
        :return: 输入矩阵的梯度
        """
        grad_X = grad_outputs[0]  # 获取梯度
        S, S_log, U = ctx.saved_tensors  # 恢复前向传播中保存的变量

        # 计算特征值和特征向量的梯度
        grad_S = S.inverse().mm(U.t().mm(sym_op(grad_X).mm(U)))  # 特征值梯度
        grad_U = 2 * sym_op(grad_X).mm(U.mm(S_log))  # 特征向量梯度

        return grad_S, grad_U


def ReEig(X, eps):
    S, U = EigDecomp.apply(X)#S是特征值对角矩阵，U是特征向量矩阵
    S_th = torch.max(eps * torch.eye(S.shape[0]).to(S.device), S)#比较对角线元素和eps*I的大小，取较大者，约等于relu
    return U.mm(S_th.mm(U.t()))
# def ReEig(X, eps):
#     """
#     Regularizes the eigenvalues of an SPD matrix by ensuring they are above a certain threshold.
#     :param X: Input SPD matrix
#     :param eps: Threshold for eigenvalues
#     :return: Regularized SPD matrix
#     """
#     S, U = torch.linalg.eigh(X)  # 特征值分解
#     S_th = torch.max(eps * torch.eye(S.shape[0]).to(S.device), S)  # 确保特征值大于阈值
#     return U @ torch.diag(S_th) @ U.t()  # 重构 SPD 矩阵

def LogEig(X):
    S, U = EigDecomp.apply(X)#S是特征值对角矩阵，U是特征向量矩阵
    X = EigenLog.apply(S, U)
    return X
