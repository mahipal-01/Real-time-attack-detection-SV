#pragma once

#include <vector>
#include <string>
#include <utility>
#include <cmath>
#include <onnxruntime_c_api.h>

class OnnxModel {
public:
    OnnxModel(const std::string& model_path, int num_layers = 2, int d_model = 64, int num_features = 49);
    ~OnnxModel();

    void reset();
    std::pair<int, float> predict(const float features[], int num_features);

private:
    const OrtApi* ort_ = nullptr;
    OrtEnv* env_ = nullptr;
    OrtSession* session_ = nullptr;
    OrtMemoryInfo* memory_info_ = nullptr;
    OrtAllocator* allocator_ = nullptr;

    std::vector<float> h_states_;
    int nl_;
    int dm_;
    int num_features_;

    OrtValue* input_tensors_[2];
    const char* input_names_[2];
    const char* output_names_[2];
};
