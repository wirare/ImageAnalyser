/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Loader.cpp                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: wirare <wirare@42angouleme.fr>             +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/07/10 17:34:31 by wirare            #+#    #+#             */
/*   Updated: 2025/07/12 02:25:25 by wirare           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */
#include <Loader.hpp>
#include <Layer.hpp>
#include <filesystem>
namespace fs = std::filesystem;

std::vector<float> preprocess_image(const std::string& path, int target_width = 128, int target_height = 128) {
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) throw std::runtime_error("Failed to load image: " + path);

    cv::resize(img, img, cv::Size(target_width, target_height));
    img.convertTo(img, CV_32F, 1.0 / 255.0);  // Normalize to [0, 1]

    std::vector<float> flat;
    flat.assign(img.begin<float>(), img.end<float>());
    return flat;
}

Tensor<float, 2> load_image_batch(const std::vector<std::string>& image_paths, size_t batch_size, int width, int height)
{
    if (image_paths.size() < batch_size)
        throw std::runtime_error("Not enough images to load a full batch");

    size_t input_dim = width * height;
    Tensor<float, 2> batch_tensor({batch_size, input_dim});

    for (size_t i = 0; i < batch_size; ++i) {
        std::vector<float> img_vec = preprocess_image(image_paths[i], width, height);
        if (img_vec.size() != input_dim)
            throw std::runtime_error("Preprocessed image has incorrect size");

        for (size_t j = 0; j < input_dim; ++j)
            batch_tensor(i, j) = img_vec[j];
    }

    return batch_tensor;
}

std::vector<std::string> get_image_paths_from_folder(const std::string& folder_path, int nb_elem)
{
	static std::set<std::string> used;

	std::vector<std::string> images;
	int i = 0;
	std::string	entry_path;

	for (const auto & entry : fs::directory_iterator(folder_path))
	{
		entry_path = entry.path();
		if (used.find(entry_path) == used.end())
		{
			used.insert(entry_path);
			images.push_back(entry_path);
			i++;
		}
		if (i == nb_elem)
			return images;
	}
	return images;
}

std::vector<float> get_labels_from_img_path(const std::vector<std::string>& images_path, const std::string& cat_path)
{
	size_t size = images_path.size();

	std::vector<float> labels(size);
	int	found = 0;

	for (const auto & path : images_path)
	{
		for (const auto & entry : fs::directory_iterator(cat_path))
		{
			auto entry_path = entry.path();
			if (entry_path == path)
			{
				labels.push_back(1.f);
				found = 1;
				break;
			}
		}
		if (!found)
			labels.push_back(0.f);
	}
	return labels;
}

int main() 
{
	std::vector<std::string> paths = get_image_paths_from_folder("Flag");
	Tensor2 X_batch = load_image_batch(paths);
}
