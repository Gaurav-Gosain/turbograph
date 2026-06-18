// Package quant implements TurboQuant, a data-oblivious vector quantizer with
// near-optimal distortion (Zandieh, Daliri, Hadian, Mirrokni, 2025;
// arXiv:2504.19874).
//
// The construction has three parts:
//
//   - A randomized Hadamard rotation spreads a vector's energy evenly across
//     coordinates, so the coordinates of a unit vector become approximately
//     i.i.d. Gaussian. The rotation is orthonormal and preserves inner products
//     exactly, which is what lets a query stay in full precision while the
//     database is quantized (asymmetric distance computation).
//   - An optimal per-coordinate scalar quantizer, the Lloyd-Max codebook for the
//     standard normal, encodes the rotated coordinates. The norm is stored
//     separately.
//   - A 1-bit QJL sketch of the quantization residual corrects the bias that the
//     MSE-optimal quantizer would otherwise introduce into inner-product
//     estimates.
//
// Two families of estimators are exposed because they trade off differently. The
// Score family is the low-variance main term, best for ranking and candidate
// generation. The IP/L2/Cosine family adds the residual correction for unbiased
// magnitudes at the cost of the sketch's variance. See the Query type for
// details.
package quant
