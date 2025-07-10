/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Dense.cpp                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: ellanglo <ellanglo@42angouleme.fr>         +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/07/09 22:18:02 by ellanglo          #+#    #+#             */
/*   Updated: 2025/07/09 23:23:13 by ellanglo         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */
#include "Tensorium/Functionnal/FunctionnalRG.hpp"
#include <Dense.hpp>
#include <immintrin.h>

typedef __m256 reg;

reg fast_exp_simd(reg x) 
{
    const reg one = _mm256_set1_ps(1.0f);
    const reg c1  = _mm256_set1_ps(1.0f);
    const reg c2  = _mm256_set1_ps(1.0f / 2.0f);
    const reg c3  = _mm256_set1_ps(1.0f / 6.0f);
    const reg c4  = _mm256_set1_ps(1.0f / 24.0f);
    const reg c5  = _mm256_set1_ps(1.0f / 120.0f);
    const reg c6  = _mm256_set1_ps(1.0f / 720.0f);
    const reg c7  = _mm256_set1_ps(1.0f / 5040.0f);

    reg x2 = _mm256_mul_ps(x, x);         // x^2
    reg x3 = _mm256_mul_ps(x2, x);        // x^3
    reg x4 = _mm256_mul_ps(x3, x);        // x^4
    reg x5 = _mm256_mul_ps(x4, x);        // x^5
    reg x6 = _mm256_mul_ps(x5, x);        // x^6
    reg x7 = _mm256_mul_ps(x6, x);        // x^7

    reg result = one;
    result = _mm256_add_ps(result, x);                        // + x
    result = _mm256_add_ps(result, _mm256_mul_ps(c2, x2));    // + x^2/2
    result = _mm256_add_ps(result, _mm256_mul_ps(c3, x3));    // + x^3/6
    result = _mm256_add_ps(result, _mm256_mul_ps(c4, x4));    // + x^4/24
    result = _mm256_add_ps(result, _mm256_mul_ps(c5, x5));    // + x^5/120
    result = _mm256_add_ps(result, _mm256_mul_ps(c6, x6));    // + x^6/720
    result = _mm256_add_ps(result, _mm256_mul_ps(c7, x7));    // + x^7/5040

    return result;
}

inline reg sigma_simd(reg x)
{
	reg	one = _mm256_set1_ps(1);
	reg	denominator = one;
	reg	exp_min_x = fast_exp_simd(_mm256_mul_ps(x, _mm256_set1_ps(-1)));
	denominator = _mm256_add_ps(denominator, exp_min_x);

	return _mm256_div_ps(one, denominator);
}

inline float sigma_scl(float x)
{
	return 1 / (1 + exp(-x));
}

inline reg swish_simd(reg x)
{	
	return _mm256_mul_ps(x, sigma_simd(x));
}

inline float swish_scl(float x)
{
	return x * sigma_scl(x);
}

inline void apply_swish(Tensor2& T)
{
	Matrix<float> T_mat = tensor_to_matrix(T);

	T_mat = T_mat.foreach(swish_simd, swish_scl);

	T = matrix_to_tensor(T_mat);
}

inline reg swish_grad_simd(reg x)
{
	reg sig = sigma_simd(x);

	reg lhs = _mm256_mul_ps(x, sig);
	reg rhs = _mm256_sub_ps(_mm256_set1_ps(1), sig);

	reg res = _mm256_mul_ps(lhs, rhs);

	return _mm256_add_ps(sig, res);
}

inline float swish_grad_scl(float x)
{
	float sig = sigma_scl(x);
	return (sig + (x * sig) * (1 - sig));
}

inline void swish_grad(Tensor2& T)
{
	Matrix<float> T_mat = tensor_to_matrix(T);

	T_mat = T_mat.foreach(swish_grad_simd, swish_grad_scl);

	T = matrix_to_tensor(T_mat);
}
