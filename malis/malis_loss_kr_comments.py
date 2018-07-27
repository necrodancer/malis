# 패키지 임포트
import tensorflow as tf # 텐서플로
import numpy as np # 넘파이
from .malis import nodelist_like, malis_loss_weights # nodelist_like는 어피니티 그래프의 에지 리스트와 일치하는 리스 생성함수, malis_loss_weights 함수는 실질적인 loss 계산 함수인 것 같다.

# MALIS 클래스
class MalisWeights(object):
    # 생성
    def __init__(self, output_shape, neighborhood): # 출력 형상과 근방을 입력으로 받는다.
        # numpy array로 캐스팅한다.
        self.output_shape = np.asarray(output_shape) 
        self.neighborhood = np.asarray(neighborhood)
        self.edge_list = nodelist_like(self.output_shape, self.neighborhood) # 출력 형상과 근방값을 입력으로 넣어 에지 리스트를 생
    
    # 에지의 가중치를 계산하는 함수.
    def get_edge_weights(self, affs, gt_affs, gt_seg, gt_aff_mask, gt_seg_unlabelled): # 입력은 어피니티, 정답_어피니티, 정답_세그먼테이션, 정답_어피니티 마스크, 정답_세그먼테이션_언레이블드
        # 레이블링되지 않은 객체 영역을 하나의 유니크한 ID로 대체
        # replace the unlabelled-object area with a new unique ID
        if gt_seg_unlabelled.size > 0:
            gt_seg[gt_seg_unlabelled == 0] = gt_seg.max() + 1

        assert affs.shape[0] == len(self.neighborhood) # 이 조건이 맞지 않으면 에러를 출력. 어피니티의 첫 형상이 근방의 길이와 같으면 OK.

        weights_neg = self.malis_pass(affs, gt_affs, gt_seg, gt_aff_mask, pos=0) # negative pass to update weights
        weights_pos = self.malis_pass(affs, gt_affs, gt_seg, gt_aff_mask, pos=1) # positive pass to update weights

        return weights_neg + weights_pos # 두 pass의 weights를 합산하고 반환.

    # constrained MALIS는 두 pass의 가중치 계산을 하도록 되어 있다. 이를 위한 함수.
    def malis_pass(self, affs, gt_affs, gt_seg, gt_aff_mask, pos): 

        # create a copy of the affinities and change them, such that in the 어피니티의 복사본을 만들고 pass 별로 어피니티 내부 값을 바꾸는 코드
        #   positive pass (pos == 1): affs[gt_affs == 0] = 0
        #   negative pass (pos == 0): affs[gt_affs == 1] = 1
        pass_affs = np.copy(affs) # 복사

        if gt_aff_mask.size == 0: # 예외처리, 마스크 크기가 0일 경우
            constraint_edges = gt_affs == (1 - pos) # 하나의 값만 변경
        else: # 마스크 크기가 0이 아니면
            constraint_edges = np.logical_and( # 변경할 인덱스 계산
                gt_affs == (1 - pos),
                gt_aff_mask == 1)
        pass_affs[constraint_edges] = (1 - pos) # 어피너티 값 변경

        weights = malis_loss_weights( # 가중치들은 모두 1차원 행렬로 변환
            gt_seg.astype(np.uint64).flatten(),
            self.edge_list[0].flatten(),
            self.edge_list[1].flatten(),
            pass_affs.astype(np.float32).flatten(),
            pos)

        weights = weights.reshape((-1,) + tuple(self.output_shape)) # 가중치를 모두 열벡터로 바꾼 후 output_shape를 다 더해준다. 왜 다 더해줄까?
        assert weights.shape[0] == len(self.neighborhood) # 가중치의 첫 차원의 크기가 근방의 크기와 같아야 한다.

        # '1-pos' samples don't contribute in the 'pos' pass
        # 포지티브 패스에서는 1-pass인 샘플들이 기여하지 않도록 한다.
        weights[gt_affs == (1 - pos)] = 0

        # masked-out samples don't contribute
        # 마스크 바깥의 샘플들도 기여하지 않도록 한다.
        if gt_aff_mask.size > 0:
            weights[gt_aff_mask == 0] = 0

        # normalize 가중치를 정규화한다.
        weights = weights.astype(np.float32)
        num_pairs = np.sum(weights)
        if num_pairs > 0:
            weights = weights/num_pairs

        return weights # 가중치 반환


# 멜리스 가중치 계산 함수
def malis_weights_op(
        affs,
        gt_affs,
        gt_seg,
        neighborhood,
        gt_aff_mask=None,
        gt_seg_unlabelled=None,
        name=None):

    '''Returns a tensorflow op to compute just the weights of the MALIS loss. 멜리스 로스의 가중치만 계산하는 텐서플로 오퍼레이션을 반환한다.
    This is to be multiplied with an edge-wise base loss and summed up to create
    the final loss. For the Euclidean loss, use ``malis_loss_op``.

    Args:

        affs (Tensor): The predicted affinities.

        gt_affs (Tensor): The ground-truth affinities.

        gt_seg (Tensor): The corresponding segmentation to the ground-truth
            affinities. Label 0 denotes background.

        neighborhood (Tensor): A list of spatial offsets, defining the
            neighborhood for each voxel.

        gt_aff_mask (Tensor): A binary mask indicating where ground-truth
            affinities are known (known = 1, unknown = 0). This is to be used
            for sparsely labelled ground-truth. Edges with unknown affinities
            will not be constrained in the two malis passes, and will not
            contribute to the loss.

        gt_seg_unlabelled (Tensor): A binary mask indicating where the
            ground-truth contains unlabelled objects (labelled = 1, unlabelled
            = 0). This is to be used for ground-truth where only some objects
            have been labelled. Note that this mask is a complement to
            ``gt_aff_mask``: It is assumed that no objects cross from labelled
            to unlabelled, i.e., the boundary is a real object boundary.
            Ground-truth affinities within the unlabelled areas should be
            masked out in ``gt_aff_mask``. Ground-truth affinities between
            labelled and unlabelled areas should be zero in ``gt_affs``.

        name (string, optional): A name to use for the operators created.

    Returns:

        A tensor with the shape of ``affs``, with MALIS weights stored for each
        edge. 멜리스 가중치를 보관하는 어피니티 형상의 텐서를 반환한다.
    '''

    if gt_aff_mask is None:
        gt_aff_mask = tf.zeros((0,))
    if gt_seg_unlabelled is None:
        gt_seg_unlabelled = tf.zeros((0,))

    output_shape = gt_seg.get_shape().as_list()

    malis_weights = MalisWeights(output_shape, neighborhood)
    malis_functor = lambda \
            affs, \
            gt_affs, \
            gt_seg, \
            gt_aff_mask, \
            gt_seg_unlabelled, \
            mw=malis_weights: \
        mw.get_edge_weights(
            affs,
            gt_affs,
            gt_seg,
            gt_aff_mask,
            gt_seg_unlabelled)

    weights = tf.py_func(
        malis_functor,
        [affs, gt_affs, gt_seg, gt_aff_mask, gt_seg_unlabelled],
        [tf.float32],
        name=name)

    return weights[0]

# 멜리스 로스 연산 함수다.
def malis_loss_op( 
        affs,
        gt_affs,
        gt_seg,
        neighborhood,
        gt_aff_mask=None,
        gt_seg_unlabelled=None,
        name=None):

    '''Returns a tensorflow op to compute the constrained MALIS loss, using the 제한된 멜리스 로스를 계산하는 텐서플로 오퍼레이션을 반환한다.
    squared distance to the target values for each edge as base loss. 이때 기본 로스로 각 에지 별 타겟값과 제곱차를 이용한다.

    In the simplest case, you need to provide predicted affinities (``affs``), 가장 단순한 케이스는, 당신은 예측된 어피니티 affs가 필요하다.
    ground-truth affinities (``gt_affs``), a ground-truth segmentation 그리고 정답 어피니티인 gt_affs도 필요하다. 정답 segmenation gt_seg도 필요하다.
    (``gt_seg``), and the neighborhood that corresponds to the affinities. 마지막으로 어피니티와 일치하는 근방도 필요하다. 근데 근방은 또 뭐지?

    This loss also supports masks indicating unknown ground-truth. We 이 로스는 알려지지않은 정답을 가리키는 마스크도 지원합니다.
    distinguish two types of unknowns: 우리는 알려지지 않은 것의 두가지 타입을 구분했어요.

        1. Out of ground-truth. This is the case at the boundary of your 첫째, 정답 범위 밖입니다. 아마 당신이 레이블링한 영역의 경계에 해당할 겁니다.
           labelled area. It is unknown whether objects continue or stop at the 레이블된 영역을 통과하거나 멈출 때 그것이 계속 연속적인지는 알 수 없습니다.
           transition of the labelled area. This mask is given on edges as 이 마스크가 에지들에 gt_aff_mask 인자로써 주어집니다.
           argument ``gt_aff_mask``.

        2. Unlabelled objects. It is known that there exists a boundary between 둘째, 레이블링 안 된 객체들. 레이블된 영역과 안된 영역들이 있을 겁니다.
           the labelled area and unlabelled objects. Withing the unlabelled 레이블 안된 객체 영역 안에서는 경계를 알 수가 없습니다. 이 마스크는 또한
           objects area, it is unknown where boundaries are. This mask is also 'gt_aff_mask' 인자로 주어집니다. 
           given on edges as argument ``gt_aff_mask``, and with an additional 그리고 추가적인 인자인 gt_seg_unlabelled는 레이블 안된 객체들이 정답 segementation
           argument ``gt_seg_unlabelled`` to indicate where unlabelled objects 내부에 있음을 알려줍니다.
           are in the ground-truth segmentation.

    Both types of unknowns require masking edges to exclude them from the loss: 두 가지 언노운은 에지에 마스킹하는 것이 그것들을 로스에서 배제하는 것을 요구합니다.
    For "out of ground-truth", these are all edges that have at least one node 즉 정답 밖에 대해서, 정답 바깥 영역에 위치하는 단 하나의 노드에 포함된 모든 에지들이 언노운입니다.
    inside the "out of ground-truth" area. For "unlabelled objects", these are 레이블 안 된 객체에 대해서, 레이블 안 된 영역 내의 모든 노드의 모든 엣지들도 언노운입니다.
    all edges that have both nodes inside the "unlabelled objects" area.

    Args:

        affs (Tensor): The predicted affinities. 예측 어피니티, 텐서

        gt_affs (Tensor): The ground-truth affinities. 정답 어피니티, 텐서

        gt_seg (Tensor): The corresponding segmentation to the ground-truth
            affinities. Label 0 denotes background. 정답 어피니티에 대한 일치하는 segmenation. label 0은 배경. 텐서

        neighborhood (Tensor): A list of spatial offsets, defining the
            neighborhood for each voxel. 공간 오프셋의 리스트, 텐서, 각 복셀의 근방을 정의한다.

        gt_aff_mask (Tensor): A binary mask indicating where ground-truth 정답 어피니티가 노운인지 언노운인지 알려주는 바이너리 마스크, 노운이면 로스 계산에 포함되고 언노운이면 포함되지 않는다
            affinities are known (known = 1, unknown = 0). This is to be used
            for sparsely labelled ground-truth and at the borders of labelled
            areas. Edges with unknown affinities will not be constrained in the
            two malis passes, and will not contribute to the loss.

        gt_seg_unlabelled (Tensor): A binary mask indicating where the 정답 어피니티가 언레이블을 포함하는지 아닌지를 알려주는 바이너리 마스크, 레이블이면 로스 계산에 포함되고 언노운이면 비포함.
            ground-truth contains unlabelled objects (labelled = 1, unlabelled
            = 0). This is to be used for ground-truth where only some objects
            have been labelled. Note that this mask is a complement to
            ``gt_aff_mask``: It is assumed that no objects cross from labelled
            to unlabelled, i.e., the boundary is a real object boundary.
            Ground-truth affinities within the unlabelled areas should be
            masked out in ``gt_aff_mask``. Ground-truth affinities between
            labelled and unlabelled areas should be zero in ``gt_affs``.

        name (string, optional): A name to use for the operators created. 이 오퍼레이터에 지정될 이름

    Returns:

        A tensor with one element, the MALIS loss. 반환 : 멜리스 로스, 즉 단일 원소만 가진 텐서 한 개.
    '''

    weights = malis_weights_op(
        affs,
        gt_affs,
        gt_seg,
        neighborhood,
        gt_aff_mask,
        gt_seg_unlabelled,
        name)
    edge_loss = tf.square(tf.subtract(gt_affs, affs))

    return tf.reduce_sum(tf.multiply(weights, edge_loss))
