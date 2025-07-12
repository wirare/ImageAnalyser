/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: wirare <wirare@42angouleme.fr>             +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/07/12 01:46:54 by wirare            #+#    #+#             */
/*   Updated: 2025/07/12 19:14:51 by wirare           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */
#include <BinaryCrossEntropy.hpp>
#include <Dense.hpp>
#include <Sequential.hpp>
#include <SigmoidOutput.hpp>
#include <Loader.hpp>
#include <filesystem>

namespace fs = std::filesystem;

void train_model(Sequential& model,
                 const std::string& image_folder,
                 size_t batch_size,
                 size_t num_epochs,
                 size_t steps_per_epoch,
                 float learning_rate)
{
    for (size_t epoch = 0; epoch < num_epochs; ++epoch)
    {
        float epoch_loss = 0.0f;

        for (size_t step = 0; step < steps_per_epoch; ++step)
        {
            std::vector<std::string> paths = get_image_paths_from_folder(image_folder, batch_size);

            std::vector<float> labels = get_labels_from_img_path(paths, "Cat");  // Make sure this returns float labels

            Tensor2 X_batch = load_image_batch(paths, batch_size);

            Tensor2 Y_pred = model.forward(X_batch);

            std::vector<float> pred_vec(Y_pred.total_size);
            for (size_t i = 0; i < pred_vec.size(); ++i)
                pred_vec[i] = Y_pred.data[i];

            float loss = BinaryCrossEntropy::compute(pred_vec, labels);
            epoch_loss += loss;

            std::vector<float> grad_vec = BinaryCrossEntropy::gradient(pred_vec, labels);

            Tensor2 grad_loss({batch_size, 1});
            for (size_t i = 0; i < batch_size; i++)
                grad_loss(i, 0) = grad_vec[i];

            model.backward(grad_loss, learning_rate);
        }

        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "] - Loss: " << (epoch_loss / steps_per_epoch) << "\n";
    }
}

void evaluate_model(Sequential& model,
                    const std::string& test_root,
                    const std::string& positive_class,  // e.g. "dogs"
                    size_t batch_size = 64,
                    float threshold = 0.5f)
{
    // 1) Gather all test paths + labels
    std::vector<std::string> all_paths;
    std::vector<float>      all_labels;

    for (auto& entry : fs::directory_iterator(test_root)) {
        if (!entry.is_directory()) continue;
        std::string class_name = entry.path().filename().string();
        float label = (class_name == positive_class) ? 1.0f : 0.0f;

        for (auto& img : fs::directory_iterator(entry.path())) {
            if (!img.is_regular_file()) continue;
            all_paths.push_back(img.path().string());
            all_labels.push_back(label);
        }
    }

    // 2) Shuffle data in unison
    std::vector<size_t> idx(all_paths.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(std::random_device{}());
    std::shuffle(idx.begin(), idx.end(), rng);

    std::vector<std::string> paths_shuffled;
    std::vector<float>       labels_shuffled;
    paths_shuffled.reserve(all_paths.size());
    labels_shuffled.reserve(all_labels.size());
    for (auto i : idx) {
        paths_shuffled.push_back(all_paths[i]);
        labels_shuffled.push_back(all_labels[i]);
    }

    // 3) Iterate in batches and predict
    size_t total = paths_shuffled.size();
    size_t correct = 0, TP=0, TN=0, FP=0, FN=0;

    for (size_t offset = 0; offset < total; offset += batch_size) {
        size_t sz = std::min(batch_size, total - offset);
        std::vector<std::string> batch_paths(
            paths_shuffled.begin() + offset,
            paths_shuffled.begin() + offset + sz
        );
        std::vector<float> batch_labels(
            labels_shuffled.begin() + offset,
            labels_shuffled.begin() + offset + sz
        );

        Tensor2 X = load_image_batch(batch_paths, sz);
        Tensor2 Y = model.forward(X);  // shape [sz,1]

        // Compare
        for (size_t i = 0; i < sz; ++i) {
            float prob = Y(i,0);
            float pred = (prob >= threshold) ? 1.0f : 0.0f;
            float truth = batch_labels[i];
            if (pred == truth) {
                ++correct;
                if (pred == 1.0f) ++TP;
                else               ++TN;
            } else {
                if (pred == 1.0f) ++FP;
                else               ++FN;
            }
        }
    }

    float accuracy = float(correct) / float(total);
    std::cout
        << "Evaluation on " << total << " images:\n"
        << "  Accuracy: " << accuracy*100.f << "%\n"
        << "  TP=" << TP << "  TN=" << TN 
        << "  FP=" << FP << "  FN=" << FN << "\n";
}
