#include "onnx_model.h"
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cmath>

#define ORT_ABORT(expr) do { \
    OrtStatus* _st = (expr); \
    if (_st) { \
        const char* _msg = ort_->GetErrorMessage(_st); \
        ort_->ReleaseStatus(_st); \
        throw std::runtime_error(_msg ? _msg : "ONNX Runtime error"); \
    } \
} while (0)

OnnxModel::OnnxModel(const std::string& model_path, int num_layers, int d_model)
    : nl_(num_layers), dm_(d_model) {

    ort_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!ort_) throw std::runtime_error("Failed to get ONNX Runtime API");

    std::wstring wpath(model_path.begin(), model_path.end());

    ORT_ABORT(ort_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ids", &env_));

    OrtSessionOptions* opts = nullptr;
    ORT_ABORT(ort_->CreateSessionOptions(&opts));
    ORT_ABORT(ort_->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL));
    ORT_ABORT(ort_->CreateSession(env_, wpath.c_str(), opts, &session_));
    ort_->ReleaseSessionOptions(opts);

    ORT_ABORT(ort_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info_));

    input_tensors_[0] = nullptr;
    input_tensors_[1] = nullptr;
    input_names_[0] = "input";
    input_names_[1] = "h_states";
    output_names_[0] = "output";
    output_names_[1] = "new_h_states";

    reset();
}

OnnxModel::~OnnxModel() {
    if (memory_info_) ort_->ReleaseMemoryInfo(memory_info_);
    if (session_) ort_->ReleaseSession(session_);
    if (env_) ort_->ReleaseEnv(env_);
}

void OnnxModel::reset() {
    h_states_.assign(nl_ * dm_, 0.0f);
}

std::pair<int, float> OnnxModel::predict(const float features[49]) {
    // Input shape: {1, 49}
    int64_t input_shape[] = {1, 49};
    size_t input_bytes = 49 * sizeof(float);

    ORT_ABORT(ort_->CreateTensorWithDataAsOrtValue(
        memory_info_, const_cast<float*>(features), input_bytes,
        input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensors_[0]));

    // Hidden state shape: {1, nl_, dm_}
    int64_t h_shape[] = {1, static_cast<int64_t>(nl_), static_cast<int64_t>(dm_)};
    size_t h_bytes = static_cast<size_t>(nl_) * static_cast<size_t>(dm_) * sizeof(float);

    ORT_ABORT(ort_->CreateTensorWithDataAsOrtValue(
        memory_info_, h_states_.data(), h_bytes,
        h_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensors_[1]));

    OrtValue* output_tensors[2] = {nullptr, nullptr};

    ORT_ABORT(ort_->Run(session_, nullptr,
        input_names_, const_cast<const OrtValue**>(input_tensors_), 2,
        output_names_, 2, output_tensors));

    // Release input tensors (owned buffers remain)
    ort_->ReleaseValue(input_tensors_[0]);
    input_tensors_[0] = nullptr;
    ort_->ReleaseValue(input_tensors_[1]);
    input_tensors_[1] = nullptr;

    // Read logits
    float* logits = nullptr;
    ORT_ABORT(ort_->GetTensorMutableData(output_tensors[0], (void**)&logits));

    // Softmax
    float max_logit = *std::max_element(logits, logits + 4);
    float sum_exp = 0.0f;
    float probs[4];
    for (int i = 0; i < 4; ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum_exp += probs[i];
    }

    if (std::isnan(sum_exp) || std::isinf(sum_exp)) {
        reset();
        return {0, 0.0f};
    }

    int pred_idx = 0;
    float max_prob = 0.0f;
    for (int i = 0; i < 4; ++i) {
        probs[i] /= sum_exp;
        if (probs[i] > max_prob) {
            max_prob = probs[i];
            pred_idx = i;
        }
    }

    // Copy new hidden state
    float* new_h = nullptr;
    ORT_ABORT(ort_->GetTensorMutableData(output_tensors[1], (void**)&new_h));
    std::copy(new_h, new_h + nl_ * dm_, h_states_.begin());

    ort_->ReleaseValue(output_tensors[0]);
    ort_->ReleaseValue(output_tensors[1]);

    return {pred_idx, max_prob};
}
