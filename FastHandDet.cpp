//
// Created by lizhang on 2023/12/4.
//

#include "FastHandDet.h"

FastHandDet::FastHandDet(const std::string& FastDetHand_engine_file_path,int input_width, int input_length, float score_threshold_, float iou_threshold_)
{
    score_threshold = score_threshold_;
    iou_threshold = iou_threshold_;
    in_w = input_width;
    in_h = input_length;

    //FastDetHand的输入张量为1x3x352x352
    this->FastHandDet_input_blob = new float[3*input_width*input_length];

    ///打开trt文件
    //打开engine二进制文件，这是通过std::ifstream类完成的，文件以二进制模式打开。
    std::ifstream FastDetHand_engine_file(FastDetHand_engine_file_path, std::ios::binary);
    //检查是否成功打开
    assert(FastDetHand_engine_file.good());

    //将文件指针移动到文件的末尾，以便获取文件的大小。
    FastDetHand_engine_file.seekg(0, std::ios::end);
    //获取了文件大小 即size
    auto size_FastDetHand_engine_file = FastDetHand_engine_file.tellg();
    //将文件指针移动回文件的开头。
    FastDetHand_engine_file.seekg(0, std::ios::beg);
    //分配一个名为trtModelStream的数组，其大小为前面获取的文件大小（size变量的值）。
    char* trtModelStream_FastDetHand = new char[size_FastDetHand_engine_file];

    //判断是否分配成功
    assert(trtModelStream_FastDetHand);

    //将文件读入trtModelStream
    FastDetHand_engine_file.read(trtModelStream_FastDetHand,  size_FastDetHand_engine_file);

    FastDetHand_engine_file.close();
    ///打开trt文件

    //初始化推理插件库
    initLibNvInferPlugins(&this->gLogger, "");

    ///步骤一:创建推理运行时(InferRuntime)对象 该对象作用如下
    //1.该函数会创建一个 TensorRT InferRuntime对象，这个对象是 TensorRT 库的核心组成部分之一。
    // InferRuntime对象是用于在推理阶段执行深度学习模型的实例。它提供了一种有效的方式来执行模型的前向传播操作。
    //2. TensorRT InferRuntime对象还负责管理 GPU 资源，包括分配和释放 GPU 内存。这对于加速推理操作非常重要，因为它可以确保有效地利用 GPU 资源，同时减少 GPU 内存泄漏的风险。
    //3. 一旦创建了 TensorRT 推理运行时对象，你可以使用它来构建、配置和执行 TensorRT 模型引擎（nvinfer1::ICudaEngine）。
    // Engine模型引擎是一个已经优化过的深度学习模型，可以高效地在 GPU 上执行推理操作。
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    ///步骤二: 通过构建好InferRuntime对象 反序列化加载引擎给到(nvinfer1::ICudaEngine对象)
    this->engine_FastDetHand = this->runtime->deserializeCudaEngine(trtModelStream_FastDetHand, size_FastDetHand_engine_file);
    assert(this->engine_FastDetHand != nullptr);

    delete[] trtModelStream_FastDetHand;

    ///步骤三:使用engine创建一个执行上下文对象(nvinfer1::IExecutionContext*)
    //执行上下文对象作用如下(nvinfer1::IExecutionContext):
    //1. 模型推理执行：执行上下文是用于执行深度学习模型推理操作的实例。
    // 一旦你创建了执行上下文，你可以使用它来加载输入数据、运行模型的前向传播，然后获取输出结果。
    // 这允许你将模型应用于实际数据，以获得推理结果。
    //2. GPU 资源管理：执行上下文还负责管理 GPU 资源，包括内存分配和释放。
    // 这确保了在执行推理操作时有效地利用 GPU 资源，同时降低了 GPU 内存泄漏的风险。
    this->context_FastDetHand = this->engine_FastDetHand->createExecutionContext();
    assert(this->context_FastDetHand != nullptr);

    //cudaStreamCreate() 是 NVIDIA CUDA 库中的函数，用于创建一个 CUDA 流（CUDA Stream）。CUDA 流是一种并行执行 CUDA 操作的方式，它允许将多个 CUDA 操作异步执行，
    // 从而提高GPU的利用率。每个CUDA流代表一个独立的任务队列，CUDA操作可以按照流的顺序在多个流之间并行执行。
    cudaStreamCreate(&this->stream);

    //engine->getNbBindings() 是用于获取 TensorRT 模型引擎（nvinfer1::ICudaEngine）绑定的输入和输出张量数量的函数。
    // 在使用TensorRT进行深度学习模型推理时，你需要知道模型引擎绑定的输入和输出张量的数量，以便为它们分配内存并正确配置推理上下文。
    //具体来说，engine->getNbBindings() 函数返回一个整数值，表示与该模型引擎相关的绑定张量的总数。这个值通常是输入张量的数量加上输出张量的数量
    this->num_bindings = this->engine_FastDetHand->getNbBindings();

    ///为初始化模型赋予初值
    for (int i = 0; i < this->num_bindings; ++i) {
        //该结构体用于保存第i个输入或输出绑定的信息
        Binding            binding;

        //一个结构体,用于表示张量(输入输出和中间层数据)
        // dims.nbDims表示维度的数量1X3X255X255 时维度为4
        // dims.d 一个整数数组，包含每个维度的大小
        //一个形状为 (batch_size, channels, height, width) (1X3X255X255)
        // 的四维图像张量可以表示为 nvinfer1::Dims 对象，其中 nbDims 为 4，d[0] 表示批量大小，d[1] 表示通道数，d[2] 表示高度，d[3] 表示宽度。
        nvinfer1::Dims     dims;

        // 在使用 TensorRT 进行深度学习模型推理时，每个模型引擎都有输入绑定和输出绑定。这些绑定指定了模型的输入和输出张量的属性，包括数据类型、维度等。
        //这些信息都是可以被获取的
        //这里是获取第i个绑定的数据类型. i=0是为输入绑定,i=1时为输出绑定
        nvinfer1::DataType dtype = this->engine_FastDetHand->getBindingDataType(i);
        //保存第i个绑定的张量的数据类型所对应的字节大小
        binding.dsize            = type_to_size(dtype);

        //获取第i个绑定的名称 i = 0 ,输入数据的名称  i = 1,输出数据的名称
        std::string        name  = this->engine_FastDetHand->getBindingName(i);
        binding.name             = name;
        std::cout<<"第"<<i<<"个绑定的名称为:"<<name<<std::endl;

        //这个函数可以判断第i个绑定是输入绑定还是输出绑定
        bool IsInput = engine_FastDetHand->bindingIsInput(i);
        if (IsInput) {
            //如果是输入绑定,将输入绑定的数量加1
            this->num_inputs += 1;
            //获取一个输入的图像张量,用dims进行保存
            dims         = this->engine_FastDetHand->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            //binding.size = get_size_by_dims(dims)计算该图像张量的所有元素的数量(1X3X255X255)
            //binding.dsize保存的元素的数据类型所需要的字节数,最终即可计算所需的总内存大小
            binding.size = get_size_by_dims(dims);
            //将该图像张量也保存进binding里面
            binding.dims = dims;
            this->input_bindings.push_back(binding);
            // set max opt shape
            // context对象用于执行推理,这里获取到输入图像张量之后,使用该函数设置推理时的输入张量维度
            this->context_FastDetHand->setBindingDimensions(i, dims);
        }
        else {
            //获取输出的张量维度信息
            dims         = this->context_FastDetHand->getBindingDimensions(i);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }

    this->make_pipe();
}

FastHandDet::~FastHandDet()
{
    this->context_FastDetHand->destroy();
    this->engine_FastDetHand->destroy();

    this->runtime->destroy();
    delete[] FastHandDet_input_blob;

//    CHECK(cudaFree(init_output));

    cudaStreamDestroy(this->stream);

    for (auto& ptr : this->FastHandDet_device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto& ptr : this->FastHandDet_host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}

void FastHandDet::make_pipe()
{
    ///为模型的每个绑定分配内存和指针
    ///device_ptrs中包含了输入和输出的GPU内存指针

    //对于输入绑定，使用 cudaMallocAsync 分配GPU内存，以便存储输入数据。
    // 它会为每个输入绑定分配一个相应大小的GPU内存块，并将指针添加到 this->device_ptrs 向量中。
    for (auto& bindings : this->input_bindings) {
        void* d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        this->FastHandDet_device_ptrs.push_back(d_ptr);
    }

    //对于输出绑定，函数分配两个内存块：一个在GPU上分配，一个在CPU上分配。
    // cudaMallocAsync 用于分配GPU内存，
    // cudaHostAlloc 用于在CPU上分配内存。
    // 然后，将这两个内存块的指针添加到 this->device_ptrs 和 this->host_ptrs 向量中。
    for (auto& bindings : this->output_bindings) {
        void * d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->FastHandDet_device_ptrs.push_back(d_ptr);
        this->FastHandDet_host_ptrs.push_back(h_ptr);
    }
}

void FastHandDet::infer(std::vector<void*> device_ptrs,std::vector<void*> host_ptrs,int num_inputs,int num_outputs,std::vector<Binding> output_bindings)
{
    // this->context->enqueueV2：使用TensorRT执行上下文 this->context 对象执行推理操作。
    // 这是实际的推理步骤，它将输入数据传递给模型并执行模型的前向传播。
    // 具体来说，enqueueV2 接受以下参数：
    // this->device_ptrs.data()：包含了输入和输出数据的GPU内存指针数组。这些指针指向了经过分配的GPU内存中的输入数据和输出数据。
    // this->stream：CUDA流，用于异步执行推理操作。
    // nullptr：这里为了简化没有传递其他回调函数。
    this->context_FastDetHand->enqueueV2(device_ptrs.data(), this->stream, nullptr);


    ///循环处理输出数据
    for (int i = 0; i < num_outputs; i++) {
        //对于每一个输出绑定,计算输出张量的大小,该大小为张量所有元素个数乘以每个元素所占用的字节数
        size_t osize = output_bindings[i].size * output_bindings[i].dsize;

        //使用 cudaMemcpyAsync 将模型的输出数据从GPU内存（this->device_ptrs[i + this->num_inputs]，其中 i 是当前输出绑定的索引）
        // 异步复制到CPU内存（this->host_ptrs[i]，其中 i 是当前输出绑定的索引）。
        //这个步骤用于将模型的输出数据从GPU内存传输到CPU内存，以便进一步处理和分析。
        CHECK(cudaMemcpyAsync(
                host_ptrs[i], device_ptrs[i + num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
    }
    //等待CUDA流中的所有操作完成,以确保在使用模型的输出数据之前，所有数据都已正确地从GPU传输到CPU。
    cudaStreamSynchronize(this->stream);
}

void FastHandDet::detect(cv::Mat im, std::vector<TargetBox> &hand_list)
{
    ///前处理
    this->image_w = im.cols;
    this->image_h = im.rows;
    //将图像进行按照模型输入大小320X240(wxh)进行缩放
    cv::Mat resizedImage;
    //这里和ncnn一样.使用的都是双线性插值算法
    cv::resize(im, resizedImage, cv::Size(in_w, in_h), 0, 0, cv::INTER_LINEAR);
//    cv::imshow("1",resizedImage);
//    cv::waitKey(2);


    //改进了循环遍历,减少了计算开销
    const int imageCols = resizedImage.cols;
    const int imageRows = resizedImage.rows;
    const int imageChannels = resizedImage.channels();
    const int imageSize = imageCols * imageRows;

    float* FastHandDetInputPtr = FastHandDet_input_blob;

    //归一化 没有减均值, 需不需要减均值,看训练的时候,有没有减均值.减均值是将数据压缩在-1-1之间.
    for (int y = 0; y < imageRows; y++) {
        for (int x = 0; x < imageCols; x++) {
            for (int c = 0; c < imageChannels; c++) {
                const int pixelIdx = c * imageSize + x + y * imageCols;
                const float pixelValue = resizedImage.at<cv::Vec3b>(y, x)[c];
                //const float pixValueDeal = (pixelValue - mean_vals[c]) * norm_vals[c];
                // 找到问题了, fastdet不需要减均值再归一,同时也不是除以256而是除以255.
                const float pixValueDeal = pixelValue/255;
                FastHandDetInputPtr[pixelIdx] = pixValueDeal;
            }
        }
    }

    ///将图像数据传入engine中
    //第一步将数据复制到给模型分配的GPU内存中,device_ptrs[0]是输入数据的GPU内存指针,device_ptrs[1] [2]是输出数据的GPU内存指针
    cudaMemcpy(FastHandDet_device_ptrs[0], FastHandDet_input_blob, input_bindings[0].size * input_bindings[0].dsize, cudaMemcpyHostToDevice);
    //第二步进行前向推理
    this->infer(FastHandDet_device_ptrs,FastHandDet_host_ptrs,num_inputs,num_outputs,output_bindings);
    //推理之后,数据已经被复制到cpu指针指向的内存中 fastDet只有一个输出 输出是6x22x22
    this->output = static_cast<float*>(this->FastHandDet_host_ptrs[0]);

    ///后处理
    std::vector<TargetBox> target_boxes;

    for (int h = 0; h < 22; h++)
    {
        for (int w = 0; w < 22; w++)
        {
            // 前景概率
            int obj_score_index = (0 * 22 * 22) + (h * 22) + w;
            float obj_score = output[obj_score_index];

            // 解析类别
            int category;
            float max_score = 0.0f;
            int class_num = 1;
            for (size_t i = 0; i < class_num; i++)
            {
                int obj_score_index = ((5 + i) * 22 * 22) + (h * 22) + w;
                float cls_score = output[obj_score_index];
                if (cls_score > max_score)
                {
                    max_score = cls_score;
                    category = i;
                }
            }
            float score = pow(max_score, 0.4) * pow(obj_score, 0.6);

            // 阈值筛选
            if(score > 0.8)
            {
                // 解析坐标
                int x_offset_index = (1 * 22 * 22) + (h * 22) + w;
                int y_offset_index = (2 * 22 * 22) + (h * 22) + w;
                int box_width_index = (3 * 22 * 22) + (h * 22) + w;
                int box_height_index = (4 * 22 * 22) + (h * 22) + w;

                float x_offset = Tanh(output[x_offset_index]);
                float y_offset = Tanh(output[y_offset_index]);
                float box_width = Sigmoid(output[box_width_index]);
                float box_height = Sigmoid(output[box_height_index]);

                float cx = (w + x_offset) / 22;
                float cy = (h + y_offset) / 22;

                int x1 = (int)((cx - box_width * 0.5) * im.size().width);
                int y1 = (int)((cy - box_height * 0.5) * im.size().height);
                int x2 = (int)((cx + box_width * 0.5) * im.size().width);
                int y2 = (int)((cy + box_height * 0.5) * im.size().height);

                target_boxes.push_back(TargetBox{x1, y1, x2, y2, category, score});
            }
        }
    }

    // NMS处理
    std::vector<TargetBox> nms_boxes;
    nmsHandle(target_boxes, nms_boxes);
    hand_list = nms_boxes;
}