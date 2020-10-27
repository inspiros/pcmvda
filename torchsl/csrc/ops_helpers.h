#pragma once

// #define TORCH_KERNEL(functor) torch::RegisterOperators::options().kernel(torch::DispatchKey::VariableTensorId, &functor)
# define TORCH_KERNEL(functor) functor

