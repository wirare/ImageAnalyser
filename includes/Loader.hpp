/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Loader.hpp                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: wirare <wirare@42angouleme.fr>             +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/07/10 17:34:34 by wirare            #+#    #+#             */
/*   Updated: 2025/07/12 02:23:29 by wirare           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */
#pragma once

#include <Tensorium/Tensorium.hpp>
#include <opencv2/opencv.hpp>
#include <Layer.hpp>

std::vector<float> get_labels_from_img_path(const std::vector<std::string>& images_path, const std::string& cat_path);
std::vector<std::string> get_image_paths_from_folder(const std::string& folder_path, int nb_elem=64);
Tensor<float, 2> load_image_batch(const std::vector<std::string>& image_paths, size_t batch_size = 64, int width = 128, int height = 128);


