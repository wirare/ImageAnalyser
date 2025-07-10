/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Dense.hpp                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: ellanglo <ellanglo@42angouleme.fr>         +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/07/09 18:13:10 by ellanglo          #+#    #+#             */
/*   Updated: 2025/07/09 23:03:17 by ellanglo         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */
#pragma once

#include <Tensorium/Tensorium.hpp>

using namespace tensorium;

typedef Tensor<float, 2> Tensor2;
typedef Tensor<float, 1> Tensor1;

void apply_swish(Tensor2& T);

class Dense {
private:
    Tensor2 W;
    Vector<float> b;
    Tensor2 X_cache;
	Tensor2 Z_cache;

public:
    Dense(size_t in_dim, size_t out_dim);

    Tensor2 forward(const Tensor2& X)
	{
		X_cache = X;
		Tensor2 Z = W * X + b;
		apply_swish(Z);
		return Z;
	}
    Tensor2 backward(const Tensor2& grad_output, float lr);
};
