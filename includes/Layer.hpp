/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Layer.hpp                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: wirare <wirare@42angouleme.fr>             +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/07/10 17:06:28 by wirare            #+#    #+#             */
/*   Updated: 2025/07/12 02:44:57 by wirare           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */
#pragma once

#include <Tensorium/Tensorium.hpp>
#include <random>

using namespace tensorium;

typedef Tensor<float, 2> Tensor2;
typedef Tensor<float, 1> Tensor1;

void apply_swish(Tensor2& T);
Tensor2 swish_grad(Tensor2& T);

class Layer {
	public:
		virtual Tensor2 forward(const Tensor2& x) = 0;
		virtual Tensor2 backward(const Tensor2& grad_output, float lr) = 0;
		virtual ~Layer() = default;

		virtual Tensor2& getWeights() = 0;
		virtual Vector<float>& getBiases() = 0;
		virtual void setWeights(const Tensor2& W) = 0;
		virtual void setBiases(const Vector<float>& b) = 0;
};
