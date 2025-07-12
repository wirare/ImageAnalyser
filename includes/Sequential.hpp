/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Sequential.hpp                                     :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: wirare <wirare@42angouleme.fr>             +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/07/10 17:21:18 by wirare            #+#    #+#             */
/*   Updated: 2025/07/12 02:46:51 by wirare           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */
#pragma once

#include <Layer.hpp>
#include <fstream>

class Sequential {
private:
    std::vector<std::unique_ptr<Layer>> layers;

public:
    void add(std::unique_ptr<Layer> layer) {
        layers.push_back(std::move(layer));
    }

    Tensor2 forward(const Tensor2& input) {
        Tensor2 out = input;
        for (auto& layer : layers) {
            out = layer->forward(out);
        }
        return out;
    }

    Tensor2 backward(const Tensor2& grad_output, float lr) {
        Tensor2 grad = grad_output;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            grad = (*it)->backward(grad, lr);
        }
        return grad;
    }

	void save_model(const Sequential& model, const std::string& filename) {
		std::ofstream ofs(filename, std::ios::binary);
		if (!ofs) throw std::runtime_error("Failed to open file for saving model");

		for (const auto& layer : model.layers) {
			auto& W = layer->getWeights();
			auto& b = layer->getBiases();

			size_t rank = W.dimensions.size();
			ofs.write(reinterpret_cast<const char*>(&rank), sizeof(rank));
			ofs.write(reinterpret_cast<const char*>(W.dimensions.data()), rank * sizeof(size_t));
			ofs.write(reinterpret_cast<const char*>(W.data.data()), W.total_size * sizeof(float));

			size_t size = b.size();
			ofs.write(reinterpret_cast<const char*>(&size), sizeof(size));
			ofs.write(reinterpret_cast<const char*>(b.getData()), b.size() * sizeof(float));
		}

		ofs.close();
	}
	void load_model(Sequential& model, const std::string& filename) {
		std::ifstream ifs(filename, std::ios::binary);
		if (!ifs) throw std::runtime_error("Failed to open file for loading model");

		for (auto& layer : model.layers) {
			size_t rank;
			ifs.read(reinterpret_cast<char*>(&rank), sizeof(rank));
			std::array<size_t, 2> dims;
			ifs.read(reinterpret_cast<char*>(dims.data()), rank * sizeof(size_t));

			Tensor<float, 2> W(dims);
			ifs.read(reinterpret_cast<char*>(W.data.data()), W.total_size * sizeof(float));

			size_t bias_size;
			ifs.read(reinterpret_cast<char*>(&bias_size), sizeof(bias_size));

			Vector<float> b(bias_size);
			ifs.read(reinterpret_cast<char*>(b.getData()), bias_size * sizeof(float));

			layer->setWeights(W);
			layer->setBiases(b);
		}

		ifs.close();
	}
};
