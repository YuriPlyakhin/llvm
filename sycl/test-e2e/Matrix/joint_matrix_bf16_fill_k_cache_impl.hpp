//------------------------------------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------------===//

#include <random>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using bfloat16 = sycl::ext::oneapi::bfloat16;

#ifndef MATRIX_SIZE
#define MATRIX_SIZE 16
#endif

#ifndef tM
#define tM 8
#endif
#ifndef tN
#define tN 16
#endif
#ifndef tK
#define tK 16
#endif

#ifndef MCACHE1
#define MCACHE1 16
#endif
#ifndef NCACHE1
#define NCACHE1 16
#endif
#ifndef KCACHE1
#define KCACHE1 16
#endif

#ifndef MCACHE2
#define MCACHE2 16
#endif
#ifndef NCACHE2
#define NCACHE2 16
#endif
#ifndef KCACHE2
#define KCACHE2 16
#endif

template <unsigned int rowsA, unsigned int colsA, unsigned int rowsB,
          unsigned int colsB, unsigned int vnniFactor, typename TOperand,
          typename TResult, unsigned int sgSize = 16>
void joint_matmul(TOperand *A, TOperand *B, TResult *C, queue &q) {
  q.submit([&](handler &h) {
    sycl::stream os {2048, 2048, h};
    h.parallel_for( // cache layer#1
        nd_range<2>{range<2>{1, 16}, range<2>{1, 16}},
        [=](nd_item<2> it) [[intel::reqd_sub_group_size(sgSize)]] {
          auto pA =
              address_space_cast<sycl::access::address_space::global_space,
                                 sycl::access::decorated::no>(A);
          // auto pB =
          //     address_space_cast<sycl::access::address_space::global_space,
          //                        sycl::access::decorated::no>(B);
          // auto pC =
          //     address_space_cast<sycl::access::address_space::global_space,
          //                        sycl::access::decorated::no>(C);
          auto m2 = it.get_group(0);
          auto n2 = it.get_group(1);
          auto m1 = it.get_local_id(0);
          auto n1 = it.get_local_id(1) / sgSize;
          auto sg = it.get_sub_group();

          // joint_matrix<sub_group, TResult, use::accumulator, tM, tN> tC[2];
          // joint_matrix_fill(sg, tC[0], 0);
          // os << "C0:";
          // joint_matrix_apply(sg, tC[0], [&](TResult &x) {os << x << " ";});
          // os << "\n";
//          joint_matrix_fill(sg, tC[1], 0);
          // os << "C1:";
          // joint_matrix_apply(sg, tC[1], [&](TResult &x) {os << x << " ";});
          // os << "\n";


          joint_matrix<sub_group, TOperand, use::a, tM, tK, layout::row_major> tA[2];
          // joint_matrix<sub_group, TOperand, use::b, tK, tN,
          //              layout::ext_intel_packed> tB;

          joint_matrix_load(sg, tA[0],
                            pA + (m2 * MCACHE2 + m1 * MCACHE1) * colsA, colsA);
          joint_matrix_load(sg, tA[1],
                            pA + (m2 * MCACHE2 + m1 * MCACHE1 + tM) * colsA,
                            colsA);
          // os << "A:";
          // joint_matrix_apply(sg, tA[0], [&](TOperand &x) {os << (int)x << " ";});
          // joint_matrix_apply(sg, tA[1], [&](TOperand &x) {os << (int)x << " ";});
          // os << "\n";

          sycl::ext::intel::experimental::matrix::joint_matrix_store(sg, tA[0],
                            pA + (m2 * MCACHE2 + m1 * MCACHE1) * colsA, colsA);
          // sycl::ext::intel::experimental::matrix::joint_matrix_store(sg, tA[1],
          //                   pA + (m2 * MCACHE2 + m1 * MCACHE1 + tM) * colsA,
          //                   colsA);

          // joint_matrix_load(sg, tB,
          //                   pB + (n2 * NCACHE2 + n1 * NCACHE1) * vnniFactor,
          //                   colsB * vnniFactor);

          // os << "B:";
          // joint_matrix_apply(sg, tB, [&](TOperand &x) {os << (int)x << " ";});
          // os << "\n";

          // sycl::ext::intel::experimental::matrix::joint_matrix_store(sg, tB,
          //                   pB + (n2 * NCACHE2 + n1 * NCACHE1) * vnniFactor,
          //                   colsB * vnniFactor);

          //joint_matrix_mad(sg, tC[0], tA[0], tB, tC[0]);

          // os << "C mad:";
          // joint_matrix_apply(sg, tC[0], [&](TResult &x) {os << (int)x << " ";});

          //joint_matrix_mad(sg, tC[1], tA[1], tB, tC[1]);

          // joint_matrix_apply(sg, tC[1], [&](TResult &x) {os << (int)x << " ";});
          // os << "\n";


          // joint_matrix_store(sg, tC[0],
          //                    pC + (m2 * MCACHE2 + m1 * MCACHE1) * colsB +
          //                        (n2 * NCACHE2 + n1 * NCACHE1),
          //                    colsB, layout::row_major);
          // joint_matrix_store(sg, tC[1],
          //                    pC + (m2 * MCACHE2 + m1 * MCACHE1 + tM) * colsB +
          //                        (n2 * NCACHE2 + n1 * NCACHE1),
          //                    colsB, layout::row_major);
        }); // parallel_for
  });       // queue.submit
  q.wait();
}

void fill_matrix(bfloat16 *M) {
  // std::random_device dev;
  // std::uniform_real_distribution<float> fdistr(-1.0, 1.0);
  bfloat16 c = 0;
  for (unsigned int i = 0; i < MATRIX_SIZE; i++) {
    for (unsigned int j = 0; j < MATRIX_SIZE; j++) {
      //M[i * MATRIX_SIZE + j] = bfloat16(fdistr(dev));
      M[i * MATRIX_SIZE + j] = c++;
    }
  }
}

void native_matmul(bfloat16 *A, bfloat16 *B, float *C) {
  memset(C, 0, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
  for (unsigned int i = 0; i < MATRIX_SIZE; i++) {
    for (unsigned int k = 0; k < MATRIX_SIZE; k++) {
      for (unsigned int j = 0; j < MATRIX_SIZE; j++) {
        C[i * MATRIX_SIZE + j] += make_fp32(A[i * MATRIX_SIZE + k]) *
                                  make_fp32(B[k * MATRIX_SIZE + j]);
      }
    }
  }
}

void print_matrix(float *C) {
      for (unsigned int j = 0; j < MATRIX_SIZE; j++) {
  for (unsigned int i = 0; i < MATRIX_SIZE; i++) {
        std::cout << C[i * MATRIX_SIZE + j] << " ";
      }
      std::cout << "\n";
  }
}

void print_matrix_diff(float *C, float *D) {
      for (unsigned int j = 0; j < MATRIX_SIZE; j++) {
  for (unsigned int i = 0; i < MATRIX_SIZE; i++) {
        std::cout << D[i * MATRIX_SIZE + j] - C[i * MATRIX_SIZE + j] << " ";
      }
      std::cout << "\n";
  }
}

void print_matrix(bfloat16 *C) {
      for (unsigned int j = 0; j < MATRIX_SIZE; j++) {
  for (unsigned int i = 0; i < MATRIX_SIZE; i++) {
        std::cout << C[i * MATRIX_SIZE + j] << " ";
      }
      std::cout << "\n";
  }
}


int main(void) {
  queue q;
  bfloat16 *A = malloc_shared<bfloat16>(MATRIX_SIZE * MATRIX_SIZE, q);
  bfloat16 *B = malloc_shared<bfloat16>(MATRIX_SIZE * MATRIX_SIZE, q);
  bfloat16 *vnniB = malloc_shared<bfloat16>(MATRIX_SIZE * MATRIX_SIZE, q);
  float *C = malloc_shared<float>(MATRIX_SIZE * MATRIX_SIZE, q);
  float *refC = malloc_shared<float>(MATRIX_SIZE * MATRIX_SIZE, q);

  // Initialize; fill matrices
  fill_matrix(A);
  fill_matrix(B);
  //matrix_vnni<bfloat16>(MATRIX_SIZE, MATRIX_SIZE, B, vnniB, 2);
  // native_matmul(A, B, refC);

  std::cout << "A before:\n";
  print_matrix(A);
  // std::cout << "B before:\n";
  // print_matrix(B);
  // std::cout << "C before:\n";
  // print_matrix(C);
  std::cout << "- Kernel start -\n";
  joint_matmul<MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, 2, bfloat16,
               float>(A, vnniB, C, q);
  std::cout << "- Kernel end -\n";
  std::cout << "A after:\n";
  print_matrix(A);
  // std::cout << "B after:\n";
  // print_matrix(B);
  // std::cout << "C after:\n";
  // print_matrix(C);
  // std::cout << "C reference:\n";
  // print_matrix(refC);
  // std::cout << "diff:\n";
  // print_matrix_diff(C, refC);

  // bool result = matrix_compare(MATRIX_SIZE, MATRIX_SIZE, C, refC);
  // std::cout << "DONE for size " << MATRIX_SIZE << std::endl;

  free(A, q);
  free(B, q);
  free(vnniB, q);
  free(C, q);
  free(refC, q);

  return 0;
}
