/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Kunal Banerjee (Intel Corp.), Dheevatsa Mudigere (Intel Corp.)
   Alexander Heinecke (Intel Corp.), Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <math.h>

#define LSTM_TIMING

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# include <mkl_service.h>
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(REAL_TYPE)
# define REAL_TYPE float
#endif

#if !defined(CHECK) && \
  (!defined(__BLAS) || (0 != __BLAS)) && /* BLAS evailable */ \
  (LIBXSMM_EQUAL(REAL_TYPE, float) || LIBXSMM_EQUAL(REAL_TYPE, double))
# define CHECK
#endif


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void init(int seed, REAL_TYPE *LIBXSMM_RESTRICT dst,
  libxsmm_blasint nrows, libxsmm_blasint ncols, libxsmm_blasint ld, double scale)
{
  const double seed1 = scale * (seed + 1);
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < ncols; ++i) {
    libxsmm_blasint j = 0;
    for (; j < nrows; ++j) {
      const libxsmm_blasint k = i * ld + j;
      dst[k] = (REAL_TYPE)(seed1 / (k + 1));
    }
    for (; j < ld; ++j) {
      const libxsmm_blasint k = i * ld + j;
      dst[k] = (REAL_TYPE)seed;
    }
  }
}


void matrix_add(libxsmm_blasint size, REAL_TYPE *a, REAL_TYPE *b, REAL_TYPE *c)
{
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i, size)
#endif
  LIBXSMM_PRAGMA_SIMD
  for (i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}


void matrix_pointwise_mult(libxsmm_blasint size, REAL_TYPE *a, REAL_TYPE *b, REAL_TYPE *c)
{
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i, size)
#endif
  LIBXSMM_PRAGMA_SIMD
  for (i = 0; i < size; i++) {
    c[i] = a[i] * b[i];
  }
}


void matrix_sigmoid(libxsmm_blasint size, REAL_TYPE *src, REAL_TYPE *dst)
{
  libxsmm_blasint i;
  REAL_TYPE exp_value;
#if defined(_OPENMP)
# pragma omp parallel for private(i, size)
#endif
  LIBXSMM_PRAGMA_SIMD
  for (i = 0; i < size; i++) {
    exp_value = (REAL_TYPE)exp( -src[i]);
    dst[i] = 1 / (1 + exp_value);
  }
}


void matrix_tanh(libxsmm_blasint size, REAL_TYPE *src, REAL_TYPE *dst)
{
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i, size)
#endif
  LIBXSMM_PRAGMA_SIMD
  for (i = 0; i < size; i++) {
    dst[i] = tanh(src[i]);
  }
}


void matrix_relu(libxsmm_blasint size, REAL_TYPE *src, REAL_TYPE *dst)
{
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i, size)
#endif
  LIBXSMM_PRAGMA_SIMD
  for (i = 0; i < size; i++) {
    dst[i] = (src[i] >= 0) ? src[i] : -src[i];
  }
}


void recursive_step(libxsmm_bgemm_handle* handle, REAL_TYPE* u, REAL_TYPE* h, REAL_TYPE* dst)
{
  libxsmm_bgemm_omp(handle, u, h, dst, 1);
}


int rnn (const libxsmm_blasint m, const libxsmm_blasint n, const libxsmm_blasint k, const libxsmm_blasint t, 
         const libxsmm_blasint bm, const libxsmm_blasint bn, const libxsmm_blasint bk, const libxsmm_bgemm_order order, const int nrepeat,
         const libxsmm_blasint b_m1, const libxsmm_blasint b_n1, const libxsmm_blasint b_k1, const libxsmm_blasint b_k2, const libxsmm_blasint b_m2,
         const libxsmm_blasint ldw, const libxsmm_blasint ldx, const libxsmm_blasint ldz, const libxsmm_blasint ldu, const libxsmm_blasint ldh)
{
#if defined(CHECK)
  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 0 : atof(env_check));
#endif
  int result = EXIT_SUCCESS;
  const double gflops = ((2.0 * m * n * k) + (2.0 * m * n * m) + (2.0 * m * n)) * t * 1E-9;
  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXSMM_GEMM_FLAGS(transa, transb);
  const REAL_TYPE alpha = 1, beta = 1;

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload target(LIBXSMM_OFFLOAD_TARGET)
#endif
  {
    REAL_TYPE* wgold = (REAL_TYPE*)libxsmm_malloc(ldw * k * sizeof(REAL_TYPE));
    REAL_TYPE* xgoldt = (REAL_TYPE*)libxsmm_malloc(ldx * n * sizeof(REAL_TYPE) * t);
    REAL_TYPE* ugold = (REAL_TYPE*)libxsmm_malloc(ldu * m * sizeof(REAL_TYPE));
    REAL_TYPE* hgold = (REAL_TYPE*)libxsmm_malloc(ldh * n * sizeof(REAL_TYPE));
    REAL_TYPE* z1gold = (REAL_TYPE*)libxsmm_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* z2gold = (REAL_TYPE*)libxsmm_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* zgold = (REAL_TYPE*)libxsmm_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* w = (REAL_TYPE*)libxsmm_malloc(m * k * sizeof(REAL_TYPE));
    REAL_TYPE* xt = (REAL_TYPE*)libxsmm_malloc(k * n * sizeof(REAL_TYPE) * t);
    REAL_TYPE* u = (REAL_TYPE*)libxsmm_malloc(m * m * sizeof(REAL_TYPE));
    REAL_TYPE* h = (REAL_TYPE*)libxsmm_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* z1t = (REAL_TYPE*)libxsmm_malloc(m * n * sizeof(REAL_TYPE) * t);
    REAL_TYPE* z2 = (REAL_TYPE*)libxsmm_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* z = (REAL_TYPE*)libxsmm_malloc(m * n * sizeof(REAL_TYPE));
    LIBXSMM_VLA_DECL(2, REAL_TYPE, xgold, xgoldt, ldx * n);
    LIBXSMM_VLA_DECL(2, REAL_TYPE, x, xt, k * n);
    LIBXSMM_VLA_DECL(2, REAL_TYPE, z1, z1t, m * n);
    libxsmm_bgemm_handle* handlewx = 0;
    libxsmm_bgemm_handle* handleuh = 0;
    libxsmm_bgemm_handle* handlett = 0;
    unsigned long long start;
    double duration;
    const libxsmm_gemm_prefetch_type strategy = LIBXSMM_PREFETCH_AUTO;
    handlewx = libxsmm_bgemm_handle_create(LIBXSMM_GEMM_PRECISION(REAL_TYPE),
      m, n, k, &bm, &bn, &bk, &b_m1, &b_n1, &b_k1, &b_k2, 
      &alpha, &beta, &gemm_flags, &strategy, &order);
    handleuh = libxsmm_bgemm_handle_create(LIBXSMM_GEMM_PRECISION(REAL_TYPE),
      m, n, m, &bm, &bn, &bm, &b_m1, &b_n1, &b_m1, &b_m2, 
      &alpha, &beta, &gemm_flags, &strategy, &order);
    handlett = libxsmm_bgemm_handle_create(LIBXSMM_GEMM_PRECISION(REAL_TYPE),
      m, n*t, k, &bm, &bn, &bk, &b_m1, &b_n1, &b_k1, &b_k2, 
      &alpha, &beta, &gemm_flags, &strategy, &order);

#if defined(LSTM_TIMING)
    double t_input_total = 0., t_recur_total = 0., t_add_total = 0.;
    unsigned long long t_input = 0, t_recur = 0, t_add = 0;
    double duration_input = 0., duration_recur = 0., duration_add = 0.;
#endif

    if (0 != handlewx && 0 != handleuh && 0 != handlett) {
      init(42, wgold, m, k, ldw, 1.0);
      int it;
      for (it = 0; it < t; ++it) {
        init(24, &LIBXSMM_VLA_ACCESS(2, xgold, it, 0, ldx * n), k, n, ldx, 1.0);
      }
      init(42, ugold, m, m, ldu, 1.0);
      init(24, hgold, m, n, ldh, 1.0);
      init( 0, z1gold, m, n, ldz, 1.0);
      init( 0, z2gold, m, n, ldz, 1.0);
      init( 0, zgold, m, n, ldz, 1.0);
      libxsmm_bgemm_copyin_a(handlewx, wgold, &ldw, w);
      for (it = 0; it < t; ++it) {
        libxsmm_bgemm_copyin_b(handlewx, &LIBXSMM_VLA_ACCESS(2, xgold, it, 0, ldx * n), &ldx, &LIBXSMM_VLA_ACCESS(2, x, it, 0, k * n));
      }
      libxsmm_bgemm_copyin_a(handleuh, ugold, &ldu, u);
      libxsmm_bgemm_copyin_b(handleuh, hgold, &ldh, h);
      for (it = 0; it < t; ++it) {
        libxsmm_bgemm_copyin_c(handlewx, z1gold, &ldz, &LIBXSMM_VLA_ACCESS(2, z1, it, 0, m * n));
      }
      libxsmm_bgemm_copyin_c(handleuh, z2gold, &ldz, z2);
      libxsmm_bgemm_copyin_c(handlewx, zgold, &ldz, z);
#if defined(MKL_ENABLE_AVX512)
      mkl_enable_instructions(MKL_ENABLE_AVX512);
#endif
      /* warmup OpenMP (populate thread pool) */
      libxsmm_bgemm_omp(handlewx, w, x, &LIBXSMM_VLA_ACCESS(2, z1, 0, 0, m * n), 1);
#if defined(CHECK)
      if (!LIBXSMM_FEQ(0, check)) {
        LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, wgold, &ldw, &LIBXSMM_VLA_ACCESS(2, xgold, 0, 0, ldx * n), &ldx, &beta, z1gold, &ldz);
      }
#endif
      libxsmm_gemm_print(stdout, LIBXSMM_GEMM_PRECISION(REAL_TYPE),
        &transa, &transb, &m, &n, &k, &alpha, w, &ldw, x, &ldx, &beta, &LIBXSMM_VLA_ACCESS(2, z1, 0, 0, m * n), &ldz);
      fprintf(stdout, "\n\n");
      /* warmup OpenMP (populate thread pool) */
      libxsmm_bgemm_omp(handleuh, u, h, z2, 1);
#if defined(CHECK)
      if (!LIBXSMM_FEQ(0, check)) {
        LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &m, &alpha, ugold, &ldu, hgold, &ldh, &beta, z2gold, &ldz);
      }
#endif
      libxsmm_gemm_print(stdout, LIBXSMM_GEMM_PRECISION(REAL_TYPE),
        &transa, &transb, &m, &n, &m, &alpha, u, &ldu, h, &ldh, &beta, z2, &ldz);
      fprintf(stdout, "\n\n");

      int s;
      int i;
      libxsmm_blasint nt = n*t;
      start = libxsmm_timer_tick();
      for (s = 0; s < nrepeat; ++s) {
#if defined(LSTM_TIMING)
        t_input = libxsmm_timer_tick();
#endif
        /* The following loop may be absorbed into libxsmm_lstm_omp */
        libxsmm_bgemm_omp(handlett, w, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, z1, 0, 0, m * n), 1/*nrepeat*/);
        /*LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &nt, &k, &alpha, w, &ldw, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &ldx, &beta, z1, &ldz);*/
#if defined(LSTM_TIMING)
        duration_input = libxsmm_timer_duration(t_input, libxsmm_timer_tick());
        t_input_total += duration_input;
        t_recur = libxsmm_timer_tick();
#endif
        for (i = 0; i < t-1; ++i) {
          recursive_step(handleuh, u, h, z2);
          matrix_add(m*n, &LIBXSMM_VLA_ACCESS(2, z1, i, 0, m * n), z2, z);
          matrix_relu(m*n, z, h);
        }
        recursive_step(handleuh, u, h, z2);
#if defined(LSTM_TIMING)
        duration_recur = libxsmm_timer_duration(t_recur, libxsmm_timer_tick());
        t_recur_total += duration_recur;
        t_add = libxsmm_timer_tick();
#endif
        matrix_add(m*n, &LIBXSMM_VLA_ACCESS(2, z1, t-1, 0, m * n), z2, z);
#if defined(LSTM_TIMING)
        duration_add = libxsmm_timer_duration(t_add, libxsmm_timer_tick());
        t_add_total += duration_add;
#endif
      }
      duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
      if (0 < duration) {
        fprintf(stdout, "\tLIBXSMM: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
      }
#if defined(CHECK)
      if (!LIBXSMM_FEQ(0, check)) { /* validate result against LAPACK/BLAS xGEMM */
        REAL_TYPE* ztest = 0;
        int i;
        start = libxsmm_timer_tick();
        for (s = 0; s < nrepeat; ++s) {
          for (i = 0; i < t-1; ++i) {
            LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, wgold, &ldw, &LIBXSMM_VLA_ACCESS(2, xgold, i, 0, k * n), &ldx, &beta, z1gold, &ldz);
            LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &m, &alpha, ugold, &ldu, hgold, &ldh, &beta, z2gold, &ldz);
            matrix_add(m*n, z1gold, z2gold, zgold);
            matrix_relu(m*n, zgold, hgold);
          }
          LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, wgold, &ldw, &LIBXSMM_VLA_ACCESS(2, xgold, t-1, 0, k * n), &ldx, &beta, z1gold, &ldz);
          LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &m, &alpha, ugold, &ldu, hgold, &ldh, &beta, z2gold, &ldz);
          matrix_add(m*n, z1gold, z2gold, zgold);
        }
        duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
        if (0 < duration) {
          fprintf(stdout, "\tBLAS: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
        }
        /* free memory not needed further; avoid double-free later on */
        libxsmm_free(wgold); wgold = 0; 
        libxsmm_free(xgoldt); xgoldt = 0; 
        libxsmm_free(ugold); ugold = 0;
        libxsmm_free(hgold); hgold = 0;
        libxsmm_free(z1gold); z1gold = 0;
        libxsmm_free(z2gold); z2gold = 0;
        libxsmm_free(w); w = 0; 
        libxsmm_free(xt); xt = 0;
        libxsmm_free(u); u = 0;
        libxsmm_free(h); h = 0; 
        libxsmm_free(z1t); z1t = 0;
        libxsmm_free(z2); z2 = 0;
        /* allocate C-matrix in regular format, and perform copy-out */
        ztest = (REAL_TYPE*)libxsmm_malloc(ldz * n * sizeof(REAL_TYPE));
        if (0 != ztest) {
          libxsmm_matdiff_info diff;
          libxsmm_bgemm_copyout_c(handleuh, z, &ldz, ztest);
          if (EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE(REAL_TYPE), m, n, zgold, ztest, &ldz, &ldz, &diff)) {
            fprintf(stdout, "\tdiff: L2abs=%f L2rel=%f\n", diff.l2_abs, diff.linf_abs);
            if (check < 100.0 * diff.normf_rel) {
              fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
              result = EXIT_FAILURE;
            }
          }
          libxsmm_free(ztest);
        }
      }
#endif
      libxsmm_bgemm_handle_destroy(handlewx);
      libxsmm_bgemm_handle_destroy(handleuh);
      libxsmm_bgemm_handle_destroy(handlett);
    }
    else {
      fprintf(stderr, "FAILED to create BGEMM-handle! For details retry with LIBXSMM_VERBOSE=1.\n");
      result = EXIT_FAILURE;
    }
    libxsmm_free(wgold);
    libxsmm_free(xgoldt);
    libxsmm_free(ugold);
    libxsmm_free(hgold);
    libxsmm_free(z1gold);
    libxsmm_free(z2gold);
    libxsmm_free(zgold);
    libxsmm_free(w);
    libxsmm_free(xt);
    libxsmm_free(u);
    libxsmm_free(h);
    libxsmm_free(z1t);
    libxsmm_free(z2);
    libxsmm_free(z);
#if defined(LSTM_TIMING)
  double t_total = t_input_total + t_recur_total + t_add_total;
  fprintf(stdout, "Percentage of time spent in input matrix multiplication: %lf\n", t_input_total*100.0/t_total);
  fprintf(stdout, "Percentage of time spent in recurrence matrix multiplication: %lf\n", t_recur_total*100.0/t_total);
  fprintf(stdout, "Percentage of time spent in addition: %lf\n", t_add_total*100.0/t_total);
#endif
  }
  fprintf(stdout, "Finished\n");

  return result;
}


int lstm (const libxsmm_blasint m, const libxsmm_blasint n, const libxsmm_blasint k, const libxsmm_blasint t, 
          const libxsmm_blasint bm, const libxsmm_blasint bn, const libxsmm_blasint bk, const libxsmm_bgemm_order order, const int nrepeat,
          const libxsmm_blasint b_m1, const libxsmm_blasint b_n1, const libxsmm_blasint b_k1, const libxsmm_blasint b_k2, const libxsmm_blasint b_m2,
          const libxsmm_blasint ldw, const libxsmm_blasint ldx, const libxsmm_blasint ldz, const libxsmm_blasint ldu, const libxsmm_blasint ldh)
{
#if defined(CHECK)
  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 0 : atof(env_check));
#endif
  int result = EXIT_SUCCESS;
  const double gflops = (((2.0 * m * n * k) + (2.0 * m * n * m) + (2.0 * m * n)) * 4.0 + (4.0 * m * n)) * t * 1E-9;
  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXSMM_GEMM_FLAGS(transa, transb);
  const REAL_TYPE alpha = 1, beta = 1;

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload target(LIBXSMM_OFFLOAD_TARGET)
#endif
  {
    REAL_TYPE* wigold = (REAL_TYPE*)libxsmm_malloc(ldw * k * sizeof(REAL_TYPE));
    REAL_TYPE* wfgold = (REAL_TYPE*)libxsmm_malloc(ldw * k * sizeof(REAL_TYPE));
    REAL_TYPE* wogold = (REAL_TYPE*)libxsmm_malloc(ldw * k * sizeof(REAL_TYPE));
    REAL_TYPE* wcgold = (REAL_TYPE*)libxsmm_malloc(ldw * k * sizeof(REAL_TYPE));
    REAL_TYPE* xgoldt = (REAL_TYPE*)libxsmm_malloc(ldx * n * sizeof(REAL_TYPE) * t);
    REAL_TYPE* rigold = (REAL_TYPE*)libxsmm_malloc(ldu * m * sizeof(REAL_TYPE));
    REAL_TYPE* rfgold = (REAL_TYPE*)libxsmm_malloc(ldu * m * sizeof(REAL_TYPE));
    REAL_TYPE* rogold = (REAL_TYPE*)libxsmm_malloc(ldu * m * sizeof(REAL_TYPE));
    REAL_TYPE* rcgold = (REAL_TYPE*)libxsmm_malloc(ldu * m * sizeof(REAL_TYPE));
    REAL_TYPE* hgold = (REAL_TYPE*)libxsmm_malloc(ldh * n * sizeof(REAL_TYPE));
    REAL_TYPE* i1gold = (REAL_TYPE*)libxsmm_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* i2gold = (REAL_TYPE*)libxsmm_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* f1gold = (REAL_TYPE*)libxsmm_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* f2gold = (REAL_TYPE*)libxsmm_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* o1gold = (REAL_TYPE*)libxsmm_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* o2gold = (REAL_TYPE*)libxsmm_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* c1gold = (REAL_TYPE*)libxsmm_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* c2gold = (REAL_TYPE*)libxsmm_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* igold = (REAL_TYPE*)libxsmm_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* fgold = (REAL_TYPE*)libxsmm_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* ogold = (REAL_TYPE*)libxsmm_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* cgold = (REAL_TYPE*)libxsmm_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* d1gold = (REAL_TYPE*)libxsmm_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* d2gold = (REAL_TYPE*)libxsmm_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* d0gold = (REAL_TYPE*)libxsmm_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* dgold = (REAL_TYPE*)libxsmm_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* wi = (REAL_TYPE*)libxsmm_malloc(m * k * sizeof(REAL_TYPE));
    REAL_TYPE* wf = (REAL_TYPE*)libxsmm_malloc(m * k * sizeof(REAL_TYPE));
    REAL_TYPE* wo = (REAL_TYPE*)libxsmm_malloc(m * k * sizeof(REAL_TYPE));
    REAL_TYPE* wc = (REAL_TYPE*)libxsmm_malloc(m * k * sizeof(REAL_TYPE));
    REAL_TYPE* xt = (REAL_TYPE*)libxsmm_malloc(m * n * sizeof(REAL_TYPE) * t);
    REAL_TYPE* ri = (REAL_TYPE*)libxsmm_malloc(m * m * sizeof(REAL_TYPE));
    REAL_TYPE* rf = (REAL_TYPE*)libxsmm_malloc(m * m * sizeof(REAL_TYPE));
    REAL_TYPE* ro = (REAL_TYPE*)libxsmm_malloc(m * m * sizeof(REAL_TYPE));
    REAL_TYPE* rc = (REAL_TYPE*)libxsmm_malloc(m * m * sizeof(REAL_TYPE));
    REAL_TYPE* h = (REAL_TYPE*)libxsmm_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* i1t = (REAL_TYPE*)libxsmm_malloc(m * n * sizeof(REAL_TYPE) * t);
    REAL_TYPE* i2 = (REAL_TYPE*)libxsmm_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* f1t = (REAL_TYPE*)libxsmm_malloc(m * n * sizeof(REAL_TYPE) * t);
    REAL_TYPE* f2 = (REAL_TYPE*)libxsmm_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* o1t = (REAL_TYPE*)libxsmm_malloc(m * n * sizeof(REAL_TYPE) * t);
    REAL_TYPE* o2 = (REAL_TYPE*)libxsmm_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* c1t = (REAL_TYPE*)libxsmm_malloc(m * n * sizeof(REAL_TYPE) * t);
    REAL_TYPE* c2 = (REAL_TYPE*)libxsmm_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* i = (REAL_TYPE*)libxsmm_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* f = (REAL_TYPE*)libxsmm_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* o = (REAL_TYPE*)libxsmm_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* c = (REAL_TYPE*)libxsmm_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* d1 = (REAL_TYPE*)libxsmm_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* d2 = (REAL_TYPE*)libxsmm_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* d0 = (REAL_TYPE*)libxsmm_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* d = (REAL_TYPE*)libxsmm_malloc(m * n * sizeof(REAL_TYPE));
    LIBXSMM_VLA_DECL(2, REAL_TYPE, xgold, xgoldt, ldx * n);
    LIBXSMM_VLA_DECL(2, REAL_TYPE, x, xt, k * n);
    LIBXSMM_VLA_DECL(2, REAL_TYPE, i1, i1t, m * n);
    LIBXSMM_VLA_DECL(2, REAL_TYPE, f1, f1t, m * n);
    LIBXSMM_VLA_DECL(2, REAL_TYPE, o1, o1t, m * n);
    LIBXSMM_VLA_DECL(2, REAL_TYPE, c1, c1t, m * n);
    libxsmm_bgemm_handle* handlewx = 0;
    libxsmm_bgemm_handle* handleuh = 0;
    libxsmm_bgemm_handle* handlett = 0;
    unsigned long long start;
    double duration;
    const libxsmm_gemm_prefetch_type strategy = LIBXSMM_PREFETCH_AUTO;
    handlewx = libxsmm_bgemm_handle_create(LIBXSMM_GEMM_PRECISION(REAL_TYPE),
      m, n, k, &bm, &bn, &bk, &b_m1, &b_n1, &b_k1, &b_k2, 
      &alpha, &beta, &gemm_flags, &strategy, &order);
    handleuh = libxsmm_bgemm_handle_create(LIBXSMM_GEMM_PRECISION(REAL_TYPE),
      m, n, m, &bm, &bn, &bm, &b_m1, &b_n1, &b_m1, &b_m2, 
      &alpha, &beta, &gemm_flags, &strategy, &order);
    handlett = libxsmm_bgemm_handle_create(LIBXSMM_GEMM_PRECISION(REAL_TYPE),
      m, n*t, k, &bm, &bn, &bk, &b_m1, &b_n1, &b_k1, &b_k2, 
      &alpha, &beta, &gemm_flags, &strategy, &order);

#if defined(LSTM_TIMING)
    double t_input_total = 0., t_recur_total = 0., t_add_total = 0.;
    unsigned long long t_input = 0, t_recur = 0, t_add = 0;
    double duration_input = 0., duration_recur = 0., duration_add = 0.;
#endif

    if (0 != handlewx && 0 != handleuh && 0 != handlett) {
      init(42, wigold, m, k, ldw, 1.0);
      init(42, wfgold, m, k, ldw, 1.0);
      init(42, wogold, m, k, ldw, 1.0);
      init(42, wcgold, m, k, ldw, 1.0);
      int it;
      for (it = 0; it < t; ++it) {
        init(24, &LIBXSMM_VLA_ACCESS(2, xgold, it, 0, ldx * n), k, n, ldx, 1.0);
      }
      init(42, rigold, m, m, ldu, 1.0);
      init(42, rfgold, m, m, ldu, 1.0);
      init(42, rogold, m, m, ldu, 1.0);
      init(42, rcgold, m, m, ldu, 1.0);
      init(24, hgold, m, n, ldh, 1.0);
      init(24, d0gold, m, n, ldh, 1.0);
      init( 0, i1gold, m, n, ldz, 1.0);
      init( 0, i2gold, m, n, ldz, 1.0);
      init( 0, f1gold, m, n, ldz, 1.0);
      init( 0, f2gold, m, n, ldz, 1.0);
      init( 0, o1gold, m, n, ldz, 1.0);
      init( 0, o2gold, m, n, ldz, 1.0);
      init( 0, c1gold, m, n, ldz, 1.0);
      init( 0, c2gold, m, n, ldz, 1.0);
      init( 0, igold, m, n, ldz, 1.0);
      init( 0, fgold, m, n, ldz, 1.0);
      init( 0, ogold, m, n, ldz, 1.0);
      init( 0, cgold, m, n, ldz, 1.0);
      init( 0, d1gold, m, n, ldz, 1.0);
      init( 0, d2gold, m, n, ldz, 1.0);
      init( 0, dgold, m, n, ldz, 1.0);
      libxsmm_bgemm_copyin_a(handlewx, wigold, &ldw, wi);
      libxsmm_bgemm_copyin_a(handlewx, wfgold, &ldw, wf);
      libxsmm_bgemm_copyin_a(handlewx, wogold, &ldw, wo);
      libxsmm_bgemm_copyin_a(handlewx, wcgold, &ldw, wc);
      for (it = 0; it < t; ++it) {
        libxsmm_bgemm_copyin_b(handlewx, &LIBXSMM_VLA_ACCESS(2, xgold, it, 0, ldx * n), &ldx, &LIBXSMM_VLA_ACCESS(2, x, it, 0, k * n));
      }
      libxsmm_bgemm_copyin_a(handleuh, rigold, &ldu, ri);
      libxsmm_bgemm_copyin_a(handleuh, rfgold, &ldu, rf);
      libxsmm_bgemm_copyin_a(handleuh, rogold, &ldu, ro);
      libxsmm_bgemm_copyin_a(handleuh, rcgold, &ldu, rc);
      libxsmm_bgemm_copyin_b(handleuh, hgold, &ldh, h);
      for (it = 0; it < t; ++it) {
        libxsmm_bgemm_copyin_c(handlewx, i1gold, &ldz, &LIBXSMM_VLA_ACCESS(2, i1, it, 0, m * n));
        libxsmm_bgemm_copyin_c(handlewx, f1gold, &ldz, &LIBXSMM_VLA_ACCESS(2, f1, it, 0, m * n));
        libxsmm_bgemm_copyin_c(handlewx, o1gold, &ldz, &LIBXSMM_VLA_ACCESS(2, o1, it, 0, m * n));
        libxsmm_bgemm_copyin_c(handlewx, c1gold, &ldz, &LIBXSMM_VLA_ACCESS(2, c1, it, 0, m * n));
      }
      libxsmm_bgemm_copyin_c(handleuh, igold, &ldh, i);
      libxsmm_bgemm_copyin_c(handleuh, fgold, &ldh, f);
      libxsmm_bgemm_copyin_c(handleuh, ogold, &ldh, o);
      libxsmm_bgemm_copyin_c(handleuh, cgold, &ldh, c);
      libxsmm_bgemm_copyin_c(handleuh, d1gold, &ldh, d1);
      libxsmm_bgemm_copyin_c(handleuh, d2gold, &ldh, d2);
      libxsmm_bgemm_copyin_c(handleuh, d0gold, &ldh, d0);
      libxsmm_bgemm_copyin_c(handleuh, dgold, &ldh, d);
#if defined(MKL_ENABLE_AVX512)
      mkl_enable_instructions(MKL_ENABLE_AVX512);
#endif
      /* warmup OpenMP (populate thread pool) */
      libxsmm_bgemm_omp(handlewx, wi, x, &LIBXSMM_VLA_ACCESS(2, i1, 0, 0, m * n), 1);
      libxsmm_bgemm_omp(handlewx, wf, x, &LIBXSMM_VLA_ACCESS(2, f1, 0, 0, m * n), 1);
      libxsmm_bgemm_omp(handlewx, wo, x, &LIBXSMM_VLA_ACCESS(2, o1, 0, 0, m * n), 1);
      libxsmm_bgemm_omp(handlewx, wc, x, &LIBXSMM_VLA_ACCESS(2, c1, 0, 0, m * n), 1);
#if defined(CHECK)
      if (!LIBXSMM_FEQ(0, check)) {
        LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, wigold, &ldw, &LIBXSMM_VLA_ACCESS(2, xgold, 0, 0, ldx * n), &ldx, &beta, i1gold, &ldz);
        LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, wfgold, &ldw, &LIBXSMM_VLA_ACCESS(2, xgold, 0, 0, ldx * n), &ldx, &beta, f1gold, &ldz);
        LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, wogold, &ldw, &LIBXSMM_VLA_ACCESS(2, xgold, 0, 0, ldx * n), &ldx, &beta, o1gold, &ldz);
        LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, wcgold, &ldw, &LIBXSMM_VLA_ACCESS(2, xgold, 0, 0, ldx * n), &ldx, &beta, c1gold, &ldz);
      }
#endif
      libxsmm_gemm_print(stdout, LIBXSMM_GEMM_PRECISION(REAL_TYPE),
        &transa, &transb, &m, &n, &k, &alpha, wi, &ldw, x, &ldx, &beta, &LIBXSMM_VLA_ACCESS(2, i1, 0, 0, m * n), &ldz);
      fprintf(stdout, "\n\n");
      /* warmup OpenMP (populate thread pool) */
      libxsmm_bgemm_omp(handleuh, ri, h, i2, 1);
      libxsmm_bgemm_omp(handleuh, rf, h, f2, 1);
      libxsmm_bgemm_omp(handleuh, ro, h, o2, 1);
      libxsmm_bgemm_omp(handleuh, rc, h, c2, 1);
#if defined(CHECK)
      if (!LIBXSMM_FEQ(0, check)) {
        LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &m, &alpha, rigold, &ldu, hgold, &ldh, &beta, i2gold, &ldz);
        LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &m, &alpha, rfgold, &ldu, hgold, &ldh, &beta, f2gold, &ldz);
        LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &m, &alpha, rogold, &ldu, hgold, &ldh, &beta, o2gold, &ldz);
        LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &m, &alpha, rcgold, &ldu, hgold, &ldh, &beta, c2gold, &ldz);
      }
#endif
      libxsmm_gemm_print(stdout, LIBXSMM_GEMM_PRECISION(REAL_TYPE),
        &transa, &transb, &m, &n, &m, &alpha, ri, &ldu, h, &ldh, &beta, i2, &ldz);
      fprintf(stdout, "\n\n");

      int s;
      int j;
      libxsmm_blasint nt = n*t;
      start = libxsmm_timer_tick();
      for (s = 0; s < nrepeat; ++s) {
#if defined(LSTM_TIMING)
        t_input = libxsmm_timer_tick();
#endif
        /* The following loop may be absorbed into libxsmm_lstm_omp */
        libxsmm_bgemm_omp(handlett, wi, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, i1, 0, 0, m * n), 1/*nrepeat*/);
        libxsmm_bgemm_omp(handlett, wf, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, f1, 0, 0, m * n), 1/*nrepeat*/);
        libxsmm_bgemm_omp(handlett, wo, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, o1, 0, 0, m * n), 1/*nrepeat*/);
        libxsmm_bgemm_omp(handlett, wc, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, c1, 0, 0, m * n), 1/*nrepeat*/);
#if defined(LSTM_TIMING)
        duration_input = libxsmm_timer_duration(t_input, libxsmm_timer_tick());
        t_input_total += duration_input;
        t_recur = libxsmm_timer_tick();
#endif
        recursive_step(handleuh, ri, h, i2);
        recursive_step(handleuh, rf, h, f2);
        recursive_step(handleuh, ro, h, o2);
        recursive_step(handleuh, rc, h, c2);
        matrix_add(m*n, &LIBXSMM_VLA_ACCESS(2, i1, 0, 0, m * n), i2, i);
        matrix_add(m*n, &LIBXSMM_VLA_ACCESS(2, f1, 0, 0, m * n), f2, f);
        matrix_add(m*n, &LIBXSMM_VLA_ACCESS(2, o1, 0, 0, m * n), o2, o);
        matrix_add(m*n, &LIBXSMM_VLA_ACCESS(2, c1, 0, 0, m * n), c2, c);
        matrix_relu(m*n, i, i); /*sigmoid*/
        matrix_relu(m*n, f, f); /*sigmoid*/
        matrix_relu(m*n, o, o); /*sigmoid*/
        matrix_relu(m*n, c, c); /*tanh*/
        matrix_pointwise_mult(m*n, f, d0, d1);
        matrix_pointwise_mult(m*n, i, c, d2);
        matrix_add(m*n, d1, d2, d);
        if (t > 1) {
          matrix_relu(m*n, d, d); /*tanh*/
          matrix_pointwise_mult(m*n, o, d, h);
        }
        for (j = 1; j < t-1; ++j) {
          recursive_step(handleuh, ri, h, i2);
          recursive_step(handleuh, rf, h, f2);
          recursive_step(handleuh, ro, h, o2);
          recursive_step(handleuh, rc, h, c2);
          matrix_add(m*n, &LIBXSMM_VLA_ACCESS(2, i1, j, 0, m * n), i2, i);
          matrix_add(m*n, &LIBXSMM_VLA_ACCESS(2, f1, j, 0, m * n), f2, f);
          matrix_add(m*n, &LIBXSMM_VLA_ACCESS(2, o1, j, 0, m * n), o2, o);
          matrix_add(m*n, &LIBXSMM_VLA_ACCESS(2, c1, j, 0, m * n), c2, c);
          matrix_relu(m*n, i, i); /*sigmoid*/
          matrix_relu(m*n, f, f); /*sigmoid*/
          matrix_relu(m*n, o, o); /*sigmoid*/
          matrix_relu(m*n, c, c); /*tanh*/
          matrix_pointwise_mult(m*n, f, d, d1);
          matrix_pointwise_mult(m*n, i, c, d2);
          matrix_add(m*n, d1, d2, d);
          matrix_relu(m*n, d, d); /*tanh*/
          matrix_pointwise_mult(m*n, o, d, h);
        }
        if (t > 1) {
          recursive_step(handleuh, ri, h, i2);
          recursive_step(handleuh, rf, h, f2);
          recursive_step(handleuh, ro, h, o2);
          recursive_step(handleuh, rc, h, c2);
          matrix_add(m*n, &LIBXSMM_VLA_ACCESS(2, i1, t-1, 0, m * n), i2, i);
          matrix_add(m*n, &LIBXSMM_VLA_ACCESS(2, f1, t-1, 0, m * n), f2, f);
          matrix_add(m*n, &LIBXSMM_VLA_ACCESS(2, o1, t-1, 0, m * n), o2, o);
          matrix_add(m*n, &LIBXSMM_VLA_ACCESS(2, c1, t-1, 0, m * n), c2, c);
          matrix_relu(m*n, i, i); /*sigmoid*/
          matrix_relu(m*n, f, f); /*sigmoid*/
          matrix_relu(m*n, o, o); /*sigmoid*/
          matrix_relu(m*n, c, c); /*tanh*/
          matrix_pointwise_mult(m*n, f, d, d1);
          matrix_pointwise_mult(m*n, i, c, d2);
          matrix_add(m*n, d1, d2, d);
        }
#if defined(LSTM_TIMING)
        duration_recur = libxsmm_timer_duration(t_recur, libxsmm_timer_tick());
        t_recur_total += duration_recur;
#endif
      }
      duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
      if (0 < duration) {
        fprintf(stdout, "\tLIBXSMM: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
      }
#if defined(CHECK)
      if (!LIBXSMM_FEQ(0, check)) { /* validate result against LAPACK/BLAS xGEMM */
        REAL_TYPE* dtest = 0;
        int j;
        start = libxsmm_timer_tick();
        for (s = 0; s < nrepeat; ++s) {
          for (j = 0; j < t; ++j) {
            LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, wigold, &ldw, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, k * n), &ldx, &beta, i1gold, &ldz);
            LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, wfgold, &ldw, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, k * n), &ldx, &beta, f1gold, &ldz);
            LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, wogold, &ldw, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, k * n), &ldx, &beta, o1gold, &ldz);
            LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, wcgold, &ldw, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, k * n), &ldx, &beta, c1gold, &ldz);
            LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &m, &alpha, rigold, &ldu, hgold, &ldh, &beta, i2gold, &ldz);
            LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &m, &alpha, rfgold, &ldu, hgold, &ldh, &beta, f2gold, &ldz);
            LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &m, &alpha, rogold, &ldu, hgold, &ldh, &beta, o2gold, &ldz);
            LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &m, &alpha, rcgold, &ldu, hgold, &ldh, &beta, c2gold, &ldz);
            matrix_add(m*n, i1gold, i2gold, igold);
            matrix_add(m*n, f1gold, f2gold, fgold);
            matrix_add(m*n, o1gold, o2gold, ogold);
            matrix_add(m*n, c1gold, c2gold, cgold);
            matrix_relu(m*n, igold, igold); /*sigmoid*/
            matrix_relu(m*n, fgold, fgold); /*sigmoid*/
            matrix_relu(m*n, ogold, ogold); /*sigmoid*/
            matrix_relu(m*n, cgold, cgold); /*tanh*/
            if (j == 0) {
              matrix_pointwise_mult(m*n, fgold, d0gold, d1gold);
            } else {
              matrix_pointwise_mult(m*n, fgold, dgold, d1gold);
            }
            matrix_pointwise_mult(m*n, igold, cgold, d2gold);
            matrix_add(m*n, d1gold, d2gold, dgold);
            if (j < t-1) {
              matrix_relu(m*n, dgold, dgold); /*tanh*/
              matrix_pointwise_mult(m*n, ogold, dgold, hgold);
            }
          }
        }
        duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
        if (0 < duration) {
          fprintf(stdout, "\tBLAS: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
        }
        /* free memory not needed further; avoid double-free later on */
        libxsmm_free(wigold); wigold = 0; 
        libxsmm_free(wfgold); wfgold = 0; 
        libxsmm_free(wogold); wogold = 0; 
        libxsmm_free(wcgold); wcgold = 0; 
        libxsmm_free(xgoldt); xgoldt = 0; 
        libxsmm_free(rigold); rigold = 0; 
        libxsmm_free(rfgold); rfgold = 0; 
        libxsmm_free(rogold); rogold = 0; 
        libxsmm_free(rcgold); rcgold = 0; 
        libxsmm_free(hgold); hgold = 0;
        libxsmm_free(i1gold); i1gold = 0;
        libxsmm_free(i2gold); i2gold = 0;
        libxsmm_free(f1gold); f1gold = 0;
        libxsmm_free(f2gold); f2gold = 0;
        libxsmm_free(o1gold); o1gold = 0;
        libxsmm_free(o2gold); o2gold = 0;
        libxsmm_free(c1gold); c1gold = 0;
        libxsmm_free(c2gold); c2gold = 0;
        libxsmm_free(igold); igold = 0;
        libxsmm_free(fgold); fgold = 0;
        libxsmm_free(ogold); ogold = 0;
        libxsmm_free(cgold); cgold = 0;
        libxsmm_free(d0gold); d0gold = 0;
        libxsmm_free(d1gold); d1gold = 0;
        libxsmm_free(d2gold); d2gold = 0;
        libxsmm_free(wi); wi = 0; 
        libxsmm_free(wf); wf = 0; 
        libxsmm_free(wo); wo = 0; 
        libxsmm_free(wc); wc = 0; 
        libxsmm_free(xt); xt = 0; 
        libxsmm_free(ri); ri = 0; 
        libxsmm_free(rf); rf = 0; 
        libxsmm_free(ro); ro = 0; 
        libxsmm_free(rc); rc = 0; 
        libxsmm_free(h); h = 0;
        libxsmm_free(i1); i1 = 0;
        libxsmm_free(i2); i2 = 0;
        libxsmm_free(f1); f1 = 0;
        libxsmm_free(f2); f2 = 0;
        libxsmm_free(o1); o1 = 0;
        libxsmm_free(o2); o2 = 0;
        libxsmm_free(c1); c1 = 0;
        libxsmm_free(c2); c2 = 0;
        libxsmm_free(i); i = 0;
        libxsmm_free(f); f = 0;
        libxsmm_free(o); o = 0;
        libxsmm_free(c); c = 0;
        libxsmm_free(d0); d0 = 0;
        libxsmm_free(d1); d1 = 0;
        libxsmm_free(d2); d2 = 0;
        /* allocate C-matrix in regular format, and perform copy-out */
        dtest = (REAL_TYPE*)libxsmm_malloc(ldz * n * sizeof(REAL_TYPE));
        if (0 != dtest) {
          libxsmm_matdiff_info diff;
          libxsmm_bgemm_copyout_c(handleuh, d, &ldz, dtest);
          if (EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE(REAL_TYPE), m, n, dgold, dtest, &ldz, &ldz, &diff)) {
            fprintf(stdout, "\tdiff: L2abs=%f L2rel=%f\n", diff.l2_abs, diff.linf_abs);
            if (check < 100.0 * diff.normf_rel) {
              fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
              result = EXIT_FAILURE;
            }
          }
          libxsmm_free(dtest);
        }
      }
#endif
      libxsmm_bgemm_handle_destroy(handlewx);
      libxsmm_bgemm_handle_destroy(handleuh);
      libxsmm_bgemm_handle_destroy(handlett);
    }
    else {
      fprintf(stderr, "FAILED to create BGEMM-handle! For details retry with LIBXSMM_VERBOSE=1.\n");
      result = EXIT_FAILURE;
    }
    libxsmm_free(wigold);
    libxsmm_free(wfgold);
    libxsmm_free(wogold);
    libxsmm_free(wcgold);
    libxsmm_free(xgoldt);
    libxsmm_free(rigold);
    libxsmm_free(rfgold);
    libxsmm_free(rogold);
    libxsmm_free(rcgold);
    libxsmm_free(hgold);
    libxsmm_free(i1gold);
    libxsmm_free(i2gold);
    libxsmm_free(f1gold);
    libxsmm_free(f2gold);
    libxsmm_free(o1gold);
    libxsmm_free(o2gold);
    libxsmm_free(c1gold);
    libxsmm_free(c2gold);
    libxsmm_free(igold);
    libxsmm_free(fgold);
    libxsmm_free(ogold);
    libxsmm_free(cgold);
    libxsmm_free(d0gold);
    libxsmm_free(d1gold);
    libxsmm_free(d2gold);
    libxsmm_free(dgold);
    libxsmm_free(wi);
    libxsmm_free(wf);
    libxsmm_free(wo);
    libxsmm_free(wc);
    libxsmm_free(xt);
    libxsmm_free(ri);
    libxsmm_free(rf);
    libxsmm_free(ro);
    libxsmm_free(rc);
    libxsmm_free(h);
    libxsmm_free(i1);
    libxsmm_free(i2);
    libxsmm_free(f1);
    libxsmm_free(f2);
    libxsmm_free(o1);
    libxsmm_free(o2);
    libxsmm_free(c1);
    libxsmm_free(c2);
    libxsmm_free(i);
    libxsmm_free(f);
    libxsmm_free(o);
    libxsmm_free(c);
    libxsmm_free(d0);
    libxsmm_free(d1);
    libxsmm_free(d2);
    libxsmm_free(d);
#if defined(LSTM_TIMING)
  double t_total = t_input_total + t_recur_total + t_add_total;
  fprintf(stdout, "Percentage of time spent in input matrix multiplication: %lf\n", t_input_total*100.0/t_total);
  fprintf(stdout, "Percentage of time spent in recurrence matrix multiplication: %lf\n", t_recur_total*100.0/t_total);
#endif
  }
  fprintf(stdout, "Finished\n");

  return result;
}


int main(int argc, char* argv[])
{
  const int type = (1 < argc ? atoi(argv[1]) : 0);
  const libxsmm_blasint m = (2 < argc ? atoi(argv[2]) : 1024);
  const libxsmm_blasint k = (4 < argc ? atoi(argv[4]) : m);
  const libxsmm_blasint n = (3 < argc ? atoi(argv[3]) : k);
  const libxsmm_blasint t = (5 < argc ? atoi(argv[5]) : 3);
  const libxsmm_blasint bm = (6 < argc ? atoi(argv[6]) : 32);
  const libxsmm_blasint bk = (8 < argc ? atoi(argv[8]) : bm);
  const libxsmm_blasint bn = (7 < argc ? atoi(argv[7]) : bk);
  const libxsmm_bgemm_order order = (libxsmm_bgemm_order)(9 < argc ? atoi(argv[9]) : 0);
  const int nrepeat = (10 < argc ? atoi(argv[10]) : 100);
  const libxsmm_blasint b_m1 = (11 < argc ? atoi(argv[11]) : 1);
  const libxsmm_blasint b_n1  = (12 < argc ? atoi(argv[12]) : 1);
  const libxsmm_blasint b_k1 = (13 < argc ? atoi(argv[13]) : 1);
  const libxsmm_blasint b_k2 = (14 < argc ? atoi(argv[14]) : 1);
  const libxsmm_blasint b_m2 = (15 < argc ? atoi(argv[15]) : 1);
  const libxsmm_blasint ldw = (16 < argc ? atoi(argv[16]) : m);
  const libxsmm_blasint ldx = (17 < argc ? atoi(argv[17]) : k);
  const libxsmm_blasint ldz = (18 < argc ? atoi(argv[18]) : m);
  const libxsmm_blasint ldu = (19 < argc ? atoi(argv[19]) : m);
  const libxsmm_blasint ldh = (20 < argc ? atoi(argv[20]) : m);
  int result = EXIT_SUCCESS;
  if (argc > 1 && !strncmp(argv[1], "-h", 3)) { /* check command line */
    printf("\nUsage: ./lstm [type: 0--RNN, 1--LSTM] [M] [N] [K] [time_steps] [bm] [bn] [bk] [order] [reps] [b_m1] [b_n1] [b_k1] [b_k2] [b_m2]\n\n");
    return result;
  }
  if (type == 0) {
    return rnn (m, n, k, t, bm, bn, bk, order, nrepeat, b_m1, b_n1, b_k1, b_k2, b_m2, ldw, ldx, ldz, ldu, ldh);
  } else if (type == 1) {
    return lstm(m, n, k, t, bm, bn, bk, order, nrepeat, b_m1, b_n1, b_k1, b_k2, b_m2, ldw, ldx, ldz, ldu, ldh);
  } else {
    fprintf(stdout, "Type %d currently not implemented!\n", type);
    return result;
  }
}

