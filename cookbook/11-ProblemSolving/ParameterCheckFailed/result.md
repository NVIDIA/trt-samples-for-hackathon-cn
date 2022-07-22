### count > 0 but values == nullptr
```
ERROR: 3: [network.cpp::addConvolutionNd::1004] Error Code 3: API Usage Error (Parameter check failed at: optimizer/api/network.cpp::addConvolutionNd::1004, condition: kernelWeights.values != nullptr
)
Segmentation fault (core dumped)
```

### count == 0 but values != nullptr
```
ERROR: 3: [layers.cpp::ConvolutionLayer::397] Error Code 3: API Usage Error (Parameter check failed at: optimizer/api/layers.cpp::ConvolutionLayer::397, condition: kernelWeights.count > 0 ? (kernelWeights.values != nullptr) : (kernelWeights.values == nullptr)
)
Segmentation fault (core dumped)
```
