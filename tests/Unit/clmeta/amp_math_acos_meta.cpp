// RUN: %cxxamp --keep %s -o %t.out
// RUN: %clmeta -x c -DQUERIES -c ./ff2f6b04bf6e1c2fd3ac92f55ef85b58.cl -o %t1.cl.o
// RUN: %gtest_amp -DDQUERIES %s %t1.cl.o -o %t1 && %t1

#include <amp.h>
#include <stdlib.h>
#include <iostream>
#include <amp_math.h>

using namespace concurrency;

#ifndef DQUERIES

int main(void) {
  const int vecSize = 1000;

  // Alloc & init input data
  extent<1> e(vecSize);
  array<float, 1> a(vecSize);
  array<float, 1> b(vecSize);
  array<float, 1> c(vecSize);

  array_view<float> ga(a);
  array_view<float> gb(b);
  array_view<float> gc(c);

  for (index<1> i(0); i[0] < vecSize; i++) {
    ga[i] = rand() / 1000.0f;
  }

  parallel_for_each(
    e,
    [=](index<1> idx) restrict(amp) {
    gc[idx] = fast_math::acos(ga[idx]);
  });

  for(unsigned i = 0; i < vecSize; i++) {
    gb[i] = fast_math::acos(ga[i]);
  }

  float sum = 0;
  for(unsigned i = 0; i < vecSize; i++) {
    sum += fast_math::fabs(fast_math::fabs(gc[i]) - fast_math::fabs(gb[i]));
  }
  return (sum > 0.1f);
}

#else

#include <gtest/gtest.h>    // must.

extern "C"
{
  extern int get_upper_kernel_ZZ4mainEN3_EC__019__cxxamp_trampolineEPfiiiiiiiS0_iiiiiii
  (unsigned arg_no, unsigned *group_id, unsigned *global_size, unsigned *local_size,
   unsigned *global_offset, unsigned work_dim, void const **arguments);
  extern  int get_lower_kernel_ZZ4mainEN3_EC__019__cxxamp_trampolineEPfiiiiiiiS0_iiiiiii
  (unsigned arg_no, unsigned *group_id, unsigned *global_size, unsigned *local_size,
   unsigned *global_offset, unsigned work_dim, void const **arguments);
  extern int get_argument_nr_kernel_ZZ4mainEN3_EC__019__cxxamp_trampolineEPfiiiiiiiS0_iiiiiii(void);
  extern int get_argument_dir_kernel_ZZ4mainEN3_EC__019__cxxamp_trampolineEPfiiiiiiiS0_iiiiiii(unsigned arg_no);
}

TEST(CLmetaTest, Category1) {
  unsigned x[]={0,0,0};
  unsigned y[]={100,100,0};
  unsigned z[]={100,1,0};
  unsigned w[]={0,0,0};
  int tdata  = 100;
  void const* arguments[]={(void*)&tdata, (void*)&tdata, (void*)&tdata, (void*)&tdata,
                           (void*)&tdata, (void*)&tdata, (void*)&tdata, (void*)&tdata,
                           (void*)&tdata, (void*)&tdata, (void*)&tdata, (void*)&tdata,
                           (void*)&tdata, (void*)&tdata, (void*)&tdata, (void*)&tdata};
  size_t ub =  get_upper_kernel_ZZ4mainEN3_EC__019__cxxamp_trampolineEPfiiiiiiiS0_iiiiiii(2, x, y, z, w, 3, arguments);
  size_t lb =  get_lower_kernel_ZZ4mainEN3_EC__019__cxxamp_trampolineEPfiiiiiiiS0_iiiiiii(2, x, y, z, w, 3, arguments);
  EXPECT_EQ(0, ub);
  EXPECT_EQ(0, lb);
  EXPECT_EQ(16, get_argument_nr_kernel_ZZ4mainEN3_EC__019__cxxamp_trampolineEPfiiiiiiiS0_iiiiiii());
  EXPECT_EQ(0, get_argument_dir_kernel_ZZ4mainEN3_EC__019__cxxamp_trampolineEPfiiiiiiiS0_iiiiiii(8));
  EXPECT_EQ(1, get_argument_dir_kernel_ZZ4mainEN3_EC__019__cxxamp_trampolineEPfiiiiiiiS0_iiiiiii(0));
  EXPECT_EQ(2, get_argument_dir_kernel_ZZ4mainEN3_EC__019__cxxamp_trampolineEPfiiiiiiiS0_iiiiiii(1));
}

#endif
