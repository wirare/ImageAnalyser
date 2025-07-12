/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   SigmoidOutput.hpp                                  :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: wirare <wirare@42angouleme.fr>             +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/07/11 16:14:13 by wirare            #+#    #+#             */
/*   Updated: 2025/07/12 02:46:34 by wirare           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */
#pragma once

#include <Layer.hpp>

class SigmoidOutput : public Layer
{
	protected:
		Tensor2 W;
		Vector<float> b;

	private:
		Tensor2 X_cache;
		Tensor2 Z_cache;
		Tensor2 A_cache;
	
	public:
		SigmoidOutput(size_t in_dim) : W({in_dim, 0}), b(1)
		{
			float stddev = std::sqrt(2.0f / static_cast<float>(in_dim));
			std::mt19937 rng(std::random_device{}());
			std::normal_distribution<float> dist(0.0f, stddev);

			for (auto &w : W.data)
				w = dist(rng);

			b(0) = 0.0f;
		}

		Tensor2 forward(const Tensor2& X) override
		{
			X_cache = X;
			Z_cache = X * W;
			for (size_t i = 0; i < Z_cache.dimensions[0]; ++i)
				Z_cache(i, 0) += b(0);

			A_cache = Z_cache;
			apply_swish(A_cache);

			return A_cache;
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
