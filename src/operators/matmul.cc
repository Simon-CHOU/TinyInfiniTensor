#include "operators/matmul.h"
#include "utils/operator_utils.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        // return std::nullopt;
         const auto &A = inputs[0];
        const auto &B = inputs[1];
        const auto &aDims = A->getDims();
        const auto &bDims = B->getDims();
        IT_ASSERT(aDims.size() >= 2 && bDims.size() >= 2);

        int aRank = static_cast<int>(aDims.size());
        int bRank = static_cast<int>(bDims.size());

        int aM = aDims[aRank - 2];
        int aK = aDims[aRank - 1];
        int bK = bDims[bRank - 2];
        int bN = bDims[bRank - 1];

        if (transA)
            std::swap(aM, aK);
        if (transB)
            std::swap(bK, bN);

        IT_ASSERT(aK == bK);
        m = aM;
        n = bN;
        k = aK;

        Shape batchA(aDims.begin(), aDims.end() - 2);
        Shape batchB(bDims.begin(), bDims.end() - 2);
        Shape batch = infer_broadcast(batchA, batchB);

        Shape out = batch;
        out.push_back(m);
        out.push_back(n);
        return {{out}};
    }

} // namespace infini