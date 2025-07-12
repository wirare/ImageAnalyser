/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   BinaryCrossEntropy.hpp                             :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: wirare <wirare@42angouleme.fr>             +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/07/12 01:40:38 by wirare            #+#    #+#             */
/*   Updated: 2025/07/12 02:22:56 by wirare           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */
#pragma once

#include <Layer.hpp>

class BinaryCrossEntropy {
public:
    static float compute(const std::vector<float>& y_pred, const std::vector<float>& y_true) {
        if (y_pred.size() != y_true.size())
            throw std::runtime_error("Size mismatch in BCE loss");

        float loss = 0.0f;
        const float epsilon = 1e-7f;

        for (size_t i = 0; i < y_pred.size(); ++i) {
            float p = std::clamp(y_pred[i], epsilon, 1.0f - epsilon);
            loss += - (y_true[i] * std::log(p) + (1.0f - y_true[i]) * std::log(1.0f - p));
        }

        return loss / static_cast<float>(y_pred.size());
    }

    static std::vector<float> gradient(const std::vector<float>& y_pred, const std::vector<float>& y_true) {
        if (y_pred.size() != y_true.size())
            throw std::runtime_error("Size mismatch in BCE gradient");

        std::vector<float> grad(y_pred.size());
        const float epsilon = 1e-7f;

        for (size_t i = 0; i < y_pred.size(); ++i) {
            float p = std::clamp(y_pred[i], epsilon, 1.0f - epsilon);
            grad[i] = - (y_true[i] / p) + ((1.0f - y_true[i]) / (1.0f - p));
            grad[i] /= static_cast<float>(y_pred.size());
        }

        return grad;
    }
};
