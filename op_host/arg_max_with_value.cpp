
#include "arg_max_with_value_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ArgMaxWithValueTilingData tiling;
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    int32_t data_sz = 1;
    int32_t block_dim = 8;
    const uint32_t TILE_NUM = 8;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();

    const int* dimension = attrs->GetAttrPointer<int>(0);
    const int32_t dimension_length = 0;
    const bool* keep_dims = attrs->GetAttrPointer<bool>(1);
    tiling.set_dimension(*dimension);
    tiling.set_keepDims(*keep_dims);

    for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    {
        data_sz *= x1_shape->GetStorageShape().GetDim(i);
        if (i == *dimension)
        {
            dimension_length = x1_shape->GetStorageShape().GetDim(i);
        }
        
    }
    tiling.set_size(data_sz);
    tiling.set_dimensionLength(dimension_length);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const int* dimension = attrs->GetAttrPointer<int>(0);
    const bool* keep_dims = attrs->GetAttrPointer<bool>(1);
    tiling.set_dimension(*dimension);
    tiling.set_keep_dims(*keep_dims);

    context->SetBlockDim(block_dim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    // *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class ArgMaxWithValue : public OpDef {
public:
    explicit ArgMaxWithValue(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("indice")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("values")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dimension").Int();
        this->Attr("keep_dims").AttrType(OPTIONAL).Bool(0);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910");

    }
};

OP_ADD(ArgMaxWithValue);
}
