#include "kernel_operator.h"

class kernelArgMax
{
public:
    __aicore__ inline kernelArgMax(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indice, GM_ADDR values,uint32_t totalLength, uint32_t tileNum, uint32_t dimension_length, int dimension)
    {
        //考生补充初始化代码
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        this->dimension = dimension;
        this->dimension_length = dimension_length;
        this->outBlockLength = this->blockLength / this->dimension_length;
        this->outTileNum = this->tileNum / this->dimension_length;

        ASSERT(tileNum != 0 && "tile num can not be zero!");
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        this->outTileLength = this->tileLength / this->dimension_length;
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * GetBlockIdx(), 
        this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->outBlockLength * GetBlockIdx(), 
        this->outBlockLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + this->outBlockLength * GetBlockIdx(), 
        this->outBlockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->outTileNum * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->outTileNum * sizeof(DTYPE_Y));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->outTileNum * sizeof(DTYPE_Z));
        pipe.InitBuffer(tmpBuffer1, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuffer2, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuffer3, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuffer4, this->tileLength * sizeof(DTYPE_X));
    }
    __aicore__ inline void Process()
    {
        /*
        Process函数执行主循环，每次循环中执行三个步骤：从全局内存拷贝数据到局部内存（CopyIn），计算（Compute），然后将结果从局部内存拷贝回全局内存（CopyOut）。
        */
        int32_t loopCount = this->tileNum*BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        //考生补充算子代码
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        //考生补充算子计算代码
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
 
 
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        //考生补充算子代码
        LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    //create queue for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    //create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;
    GlobalTensor<DTYPE_Z> zGm;

    //考生补充自定义成员变量
    TBuf<QuePosition::VECCALC> tmpBuffer1, tmpBuffer2, tmpBuffer3, tmpBuffer4;
    uint32_t blockLength;
    uint32_t outBlockLength;
    uint32_t tileNum;
    uint32_t outTileNum;
    uint32_t tileLength;
    uint32_t outTileLength;
    int dimension;
    uint32_t dimension_length;
};

extern "C" __global__ __aicore__ void arg_max_with_value(GM_ADDR x, GM_ADDR indice, GM_ADDR values, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    kernelArgMax op;
    op.Init(x, indice, values, tiling_data.totalLength, tiling_data.tileNum, tiling.dimensionLength, tiling.dimension);
    op.Process();
}