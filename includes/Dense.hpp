/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Dense.hpp                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: ellanglo <ellanglo@42angouleme.fr>         +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/07/09 18:13:10 by ellanglo          #+#    #+#             */
/*   Updated: 2025/07/12 02:46:21 by wirare           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */
#pragma once

#include <Layer.hpp>

void apply_swish(Tensor2& T);
Tensor2 swish_grad(Tensor2& T);

class Dense : public Layer
{
protected:
	Tensor2 W;
	Vector<float> b;

private:
    Tensor2 X_cache;
	Tensor2 Z_cache;

public: 
    Dense(size_t in_dim, size_t out_dim) : W({in_dim, out_dim}), b(out_dim)
	{
		float stddev = std::sqrt(2.0f / static_cast<float>(in_dim));
		std::mt19937 rng(std::random_device{}());
		std::normal_distribution<float> dist(0.0f, stddev);

		for (auto &w: W.data)
			w = dist(rng);

		for (auto &bias : b.data)
			bias = 0.0f;
	}

    Tensor2 forward(const Tensor2& X) override
	{
		X_cache = X;
		Z_cache = W * X + b;
		apply_swish(Z_cache);
		return Z_cache;
	}
    Tensor2 backward(const Tensor2& grad_output, float lr) override
	{
		Tensor2 dZ = grad_output * swish_grad(Z_cache);
		Tensor2 dW = X_cache.transpose_simd() * dZ;
		Vector<float> db = dZ.sum_rows();
		Tensor2 dA_prev = dZ * W.transpose_simd();

		W -= dW * lr;
		b -= db * lr;

		return dA_prev;
	}

	Tensor2& getWeights() override { return W; }
	Vector<float>& getBiases() override { return b; }
	void setWeights(const Tensor2& Weights) override { W = Weights; }
	void setBiases(const Vector<float>& Biases) override { b = Biases; }
};
