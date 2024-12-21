#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;

class kernelArgMax
{
public:
    __aicore__ inline kernelArgMax(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,uint32_t totalLength, uint32_t tileNum, uint32_t upper, uint32_t length, uint32_t lower, int dimension)
    {
        //考生补充初始化代码
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        this->dimension = dimension;
        this->upper = upper;
        this->length = length;
        this->lower = lower;

        ASSERT(tileNum != 0 && "tile num can not be zero!");
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        this->iStart = this->blockLength * GetBlockIdx();
        this->iEnd = this->iStart + this->blockLength;
        this->upperBegin = (this->iStart / this->lower / this->length) % this->upper;
        this->upperEnd = ((this->iEnd - 1) / this->lower / this->length) % this->upper;
        this->oBlockLength = (this->upperEnd - this->upperBegin + 1) * this->lower;
        this->oTileLength = this->oBlockLength / tileNum / BUFFER_NUM;


        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->iStart, this->blockLength);
        // yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->oStart, this->oBlockLength);
        // zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + this->oStart, this->oBlockLength);
        // yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + GetBlockIdx(), this->lower);
        // zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + GetBlockIdx(), this->lower);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->upperBegin*this->lower, this->oBlockLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + this->upperBegin*this->lower, this->oBlockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->oTileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->oTileLength * sizeof(DTYPE_Z));
        // pipe.InitBuffer(tmpBuffer1, this->tileLength * sizeof(DTYPE_X));
        // pipe.InitBuffer(tmpBuffer2, this->tileLength * sizeof(DTYPE_X));
        // pipe.InitBuffer(tmpBuffer3, this->tileLength * sizeof(DTYPE_X));
        // pipe.InitBuffer(tmpBuffer4, this->tileLength * sizeof(DTYPE_X));
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
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();

        for (int i = 0; i < this->oTileLength; i++)
        {
            DTYPE_Y maxIndex = -1;
            DTYPE_Z maxValue = -FLT_MAX;
            for (int k = 0; k < this->length; k++)
            {
                DTYPE_Y tempIdx = this->upperBegin*this->lower + k*this->lower + i % this->lower;
                if (tempIdx >= this->iStart && tempIdx < this->iEnd)
                {
                    DTYPE_Z tempVal = xLocal[tempIdx - this->iStart];
                }
                if (tempVal > maxValue)
                {
                    maxValue = tempVal;
                    maxIndex = k;
                }
            }
            yLocal[i] = maxIndex;
            zLocal[i] = maxValue;
        }
        


 
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();

        for (int i = 0; i < this->oTileLength; i++)
        {
            if (zLocal[i] > zGm[progress * this->oTileLength + i])
            {
                yGm[progress * this->oTileLength + i] = yLocal[i];
                zGm[progress * this->oTileLength + i] = zLocal[i];
            }
            
        }
        // DataCopy(yGm[progress * this->oTileLength], yLocal, this->oTileLength);
        // DataCopy(zGm[progress * this->oTileLength], zLocal, this->oTileLength);
        outQueueY.FreeTensor(yLocal);
        outQueueZ.FreeTensor(zLocal);
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

    // TBuf<QuePosition::VECCALC> tmpBuffer1, tmpBuffer2, tmpBuffer3, tmpBuffer4;
    uint32_t blockLength;
    uint32_t oBlockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t oTileLength;
    int dimension;
    uint32_t upper;
    uint32_t length;
    uint32_t lower;
    uint32_t iStart;
    uint32_t iEnd;
    uint32_t oStart;
    uint32_t oEnd;
    uint32_t upperBegin;
    uint32_t upperEnd;

};

extern "C" __global__ __aicore__ void arg_max_with_value(GM_ADDR x, GM_ADDR indice, GM_ADDR values, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    kernelArgMax op;
    op.Init(x, indice, values, tiling_data.totalLength, tiling_data.tileNum, tiling.upper, tiling.length, tiling.lower, tiling.dimension);
    op.Process();
}