#include <gtest/gtest.h>

#include <torch/torch.h>

#include <algorithm>
#include <memory>
#include <vector>

#include <test/cpp/api/support.h>

using namespace torch::nn;
using namespace torch::test;

struct ParameterListTest : torch::test::SeedingFixture {};

TEST_F(ParameterListTest, ConstructsFromSharedPointer) {
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));
  torch::Tensor tb = torch::randn({1, 2}, torch::requires_grad(false));
  torch::Tensor tc = torch::randn({1, 2});
  ASSERT_TRUE(ta.requires_grad());
  ASSERT_FALSE(tb.requires_grad());
  ParameterList list(ta, tb, tc);
  ASSERT_EQ(list->size(), 3);
}

TEST_F(ParameterListTest, PushBackAddsAnElement) {
  ParameterList list;
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));
  torch::Tensor tb = torch::randn({1, 2}, torch::requires_grad(false));
  torch::Tensor tc = torch::randn({1, 2});
  torch::Tensor td = torch::randn({1, 2, 3});
  ASSERT_EQ(list->size(), 0);
  ASSERT_TRUE(list->is_empty());
  list->append(ta);
  ASSERT_EQ(list->size(), 1);
  list->append(tb);
  ASSERT_EQ(list->size(), 2);
  list->append(tc);
  ASSERT_EQ(list->size(), 3);
  list->append(td);
  ASSERT_EQ(list->size(), 4);
}

TEST_F(ParameterListTest, AccessWithAt) {
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));
  torch::Tensor tb = torch::randn({1, 2}, torch::requires_grad(false));
  torch::Tensor tc = torch::randn({1, 2});
  torch::Tensor td = torch::randn({1, 2, 3});
  std::vector<torch::Tensor> params = {ta, tb, tc, td};

  ParameterList list;
  for (auto& param : params) {
    list->append(param);
  }
  ASSERT_EQ(list->size(), 4);


  // returns the correct module for a given index
  for (size_t i = 0; i < params.size(); ++i) {
    ASSERT_TRUE(torch::all(torch::eq(list->at(i), params[i])).item<bool>());
  }

  for (size_t i = 0; i < params.size(); ++i) {
    ASSERT_TRUE(torch::all(torch::eq(list[i], params[i])).item<bool>());
  }

  // throws for a bad index
  ASSERT_THROWS_WITH(list->at(params.size() + 1), "Index 5 is out of bounds");
  ASSERT_THROWS_WITH(list[ params.size() + 1 ], "Index 5 is out of bounds");
}

TEST_F(ParameterListTest, ExtendPushesParametersFromOtherParameterList) {
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));
  torch::Tensor tb = torch::randn({1, 2}, torch::requires_grad(false));
  torch::Tensor tc = torch::randn({1, 2});
  torch::Tensor td = torch::randn({1, 2, 3});
  torch::Tensor te = torch::randn({1, 2});
  torch::Tensor tf = torch::randn({1, 2, 3});
  ParameterList a(ta, tb);
  ParameterList b(tc, td);
  a->extend(*b);

  ASSERT_EQ(a->size(), 4);
  ASSERT_TRUE(torch::all(torch::eq(a[0], ta)).item<bool>());
  ASSERT_TRUE(torch::all(torch::eq(a[1], tb)).item<bool>());
  ASSERT_TRUE(torch::all(torch::eq(a[2], tc)).item<bool>());
  ASSERT_TRUE(torch::all(torch::eq(a[3], td)).item<bool>());

  ASSERT_EQ(b->size(), 2);
  ASSERT_TRUE(torch::all(torch::eq(b[0], tc)).item<bool>());
  ASSERT_TRUE(torch::all(torch::eq(b[1], td)).item<bool>());

  std::vector<torch::Tensor> c = {te, tf};
  b->extend(c);

  ASSERT_EQ(b->size(), 4);
  ASSERT_TRUE(torch::all(torch::eq(b[0], tc)).item<bool>());
  ASSERT_TRUE(torch::all(torch::eq(b[1], td)).item<bool>());
  ASSERT_TRUE(torch::all(torch::eq(b[2], te)).item<bool>());
  ASSERT_TRUE(torch::all(torch::eq(b[3], tf)).item<bool>());
}


TEST_F(ParameterListTest, PrettyPrintParameterList) {
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));
  torch::Tensor tb = torch::randn({1, 2}, torch::requires_grad(false));
  torch::Tensor tc = torch::randn({1, 2});
  ParameterList list(ta, tb, tc);
  ASSERT_EQ(
      c10::str(list),
      "torch::nn::ParameterList(\n"
      " 1\n"
      "[ CPUFloatType{1} ]\n"
      " 2\n"
      "[ CPUFloatType{1} ]\n"
      " 3\n"
      "[ CPUFloatType{1} ]\n"
      ")");
}
