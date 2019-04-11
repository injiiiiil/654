#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensorRandom.cpp"
#else

#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cpuinfo.h>
#include <ATen/Utils.h>

void THTensor_(random)(THTensor *self, at::Generator *_generator)
{
  auto gen = at::check_generator_with_default<at::CPUGenerator>(_generator, at::detail::getDefaultCPUGenerator());
  // See Note [Thread-safety and Generators]
  std::lock_guard<std::mutex> lock(gen->mutex_);
#if defined(TH_REAL_IS_BYTE)
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (uint8_t)(gen->random() % (UINT8_MAX + 1)););
#elif defined(TH_REAL_IS_CHAR)
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (int8_t)(gen->random() % (INT8_MAX + 1)););
#elif defined(TH_REAL_IS_SHORT)
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (int16_t)(gen->random() % (INT16_MAX + 1)););
#elif defined(TH_REAL_IS_INT)
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (int32_t)(gen->random() % (INT32_MAX + 1UL)););
#elif defined(TH_REAL_IS_LONG)
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (uint64_t)(gen->random64() % (LONG_MAX + 1ULL)););
#elif defined(TH_REAL_IS_FLOAT)
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (float)(gen->random() % ((1ULL << FLT_MANT_DIG) + 1)););
#elif defined(TH_REAL_IS_DOUBLE)
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (double)(gen->random64() % ((1ULL << DBL_MANT_DIG) + 1)););
#elif defined(TH_REAL_IS_BOOL)
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (bool)(gen->random() % 2););
#else
#error "Unknown type"
#endif

}

void THTensor_(clampedRandom)(THTensor *self, at::Generator *_generator, int64_t min, int64_t max) {
  THArgCheck(max > min, 2, "max must be greater than min, but got: min = %lld, max = %lld", min, max);
  uint64_t range = max - min;
  auto gen = at::check_generator_with_default<at::CPUGenerator>(_generator, at::detail::getDefaultCPUGenerator());
  // See Note [Thread-safety and Generators]
  std::lock_guard<std::mutex> lock(gen->mutex_);
#if defined(TH_REAL_IS_LONG) || defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
    if (range >= 1ULL << 32) {
      TH_TENSOR_APPLY(scalar_t, self, *self_data = static_cast<scalar_t>(static_cast<int64_t>((gen->random64() % range) + min));)
      return;
    }
#endif
    TH_TENSOR_APPLY(scalar_t, self, *self_data = static_cast<scalar_t>(static_cast<int64_t>((gen->random() % range) + min));)
}

void THTensor_(cappedRandom)(THTensor *self, at::Generator *_generator, int64_t max) {
  THArgCheck(max > 0, 1, "max must be positive, but got: max = %lld", max);
  THTensor_(clampedRandom)(self, _generator, 0, max);
}

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

#if defined(TH_REAL_IS_FLOAT)
#define TH_REAL_MIN FLT_MIN
#elif defined(TH_REAL_IS_DOUBLE)
#define TH_REAL_MIN DBL_MIN
#endif

void THTensor_(uniform)(THTensor *self, at::Generator *_generator, double a, double b)
{
  auto gen = at::check_generator_with_default<at::CPUGenerator>(_generator, at::detail::getDefaultCPUGenerator());
  // See Note [Thread-safety and Generators]
  std::lock_guard<std::mutex> lock(gen->mutex_);

  #if defined(TH_REAL_IS_FLOAT)
  at::uniform_real_distribution<float> uniform((float)a, (float)b);
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (scalar_t)uniform(gen););
  #else
  at::uniform_real_distribution<double> uniform(a, b);
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (scalar_t)uniform(gen););
  #endif
}

void THTensor_(normal)(THTensor *self, at::Generator *_generator, double mean, double stddev)
{
  const int64_t size = THTensor_(numel)(self);
  if (size >= 16 && THTensor_(isContiguous)(self)) {
    THVector_(normal_fill)(THStorage_(data)(THTensor_getStoragePtr(self)) + self->storage_offset(), size, _generator, mean, stddev);
  } else {
    auto gen = at::check_generator_with_default<at::CPUGenerator>(_generator, at::detail::getDefaultCPUGenerator());
    // See Note [Thread-safety and Generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);

    at::normal_distribution<double> normal(mean, stddev);
    TH_TENSOR_APPLY(scalar_t, self, *self_data = (scalar_t)normal(gen););
  }
}

void THTensor_(normal_means)(THTensor *self, at::Generator *gen, THTensor *means, double stddev)
{
  THTensor_(resizeAs)(self, means);
  THTensor_(normal)(self, gen, 0, stddev);
  THTensor_(cadd)(self, self, 1, means);
}

void THTensor_(normal_stddevs)(THTensor *self, at::Generator *gen, double mean, THTensor *stddevs)
{
  THTensor_(resizeAs)(self, stddevs);
  THTensor_(normal)(self, gen, 0, 1);
  THTensor_(cmul)(self, self, stddevs);
  THTensor_(add)(self, self, mean);
}

void THTensor_(normal_means_stddevs)(THTensor *self, at::Generator *gen, THTensor *means, THTensor *stddevs)
{
  THTensor_(resizeAs)(self, means);
  THTensor_(normal)(self, gen, 0, 1);
  THTensor_(cmul)(self, self, stddevs);
  THTensor_(cadd)(self, self, 1, means);
}

void THTensor_(exponential)(THTensor *self, at::Generator *_generator, double lambda)
{
  auto gen = at::check_generator_with_default<at::CPUGenerator>(_generator, at::detail::getDefaultCPUGenerator());
  // See Note [Thread-safety and Generators]
  std::lock_guard<std::mutex> lock(gen->mutex_);

  at::exponential_distribution<double> exponential(lambda);
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (scalar_t)exponential(gen););
}

#undef TH_REAL_MIN

void THTensor_(cauchy)(THTensor *self, at::Generator *_generator, double median, double sigma)
{
  auto gen = at::check_generator_with_default<at::CPUGenerator>(_generator, at::detail::getDefaultCPUGenerator());
  // See Note [Thread-safety and Generators]
  std::lock_guard<std::mutex> lock(gen->mutex_);

  at::cauchy_distribution<double> cauchy(median, sigma);
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (scalar_t)cauchy(gen););

}

void THTensor_(logNormal)(THTensor *self, at::Generator *_generator, double mean, double stdv)
{
  auto gen = at::check_generator_with_default<at::CPUGenerator>(_generator, at::detail::getDefaultCPUGenerator());
  // See Note [Thread-safety and Generators]
  std::lock_guard<std::mutex> lock(gen->mutex_);
  
  at::lognormal_distribution<double> logNormal(mean, stdv);
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (scalar_t)logNormal(gen););
}

void THTensor_(multinomialAliasSetup)(THTensor *probs, THLongTensor *J, THTensor *q)
{
  int64_t inputsize = THTensor_(nElement)(probs);
  THArgCheck(probs->dim() == 1, 1,
             "expected 1-D probability tensor, got %d-D probability tensor instead",
             probs->dim());
  int64_t i = 0;
  THLongTensor *smaller = THLongTensor_newWithSize1d(inputsize);
  THLongTensor *larger = THLongTensor_newWithSize1d(inputsize);
  int64_t small_c = 0;
  int64_t large_c = 0;
  THLongTensor_resize1d(J, inputsize);
  THTensor_(resize1d)(q, inputsize);
  scalar_t *q_data = q->data<scalar_t>();
  int64_t *J_data = THLongTensor_data(J);

  for (i = 0; i < inputsize; i++)
    {
      THLongTensor_fastSet1d(J, i, 0L);
      scalar_t val = THTensor_(fastGet1d)(probs, i);
      THTensor_(fastSet1d)(q, i, inputsize*val);

      if (inputsize * val < 1.0)
        {
          THLongTensor_fastSet1d(smaller, small_c, i);
          small_c += 1;
        }
      else
        {
          THLongTensor_fastSet1d(larger, large_c, i);
          large_c += 1;
        }
    }

  // Loop through and create little binary mixtures that
  // appropriately allocate the larger outcomes over the
  // overall uniform mixture.
  int64_t large, small;
  while (small_c > 0 && large_c > 0)
    {
      large = THLongTensor_fastGet1d(larger, large_c-1);
      small = THLongTensor_fastGet1d(smaller, small_c-1);

      THLongTensor_fastSet1d(J, small, large);
      q_data[large * q->stride(0)] -= 1.0 - THTensor_(fastGet1d)(q, small);

      if(q_data[large * q->stride(0)] < 1.0)
        {
          THLongTensor_fastSet1d(smaller, small_c-1, large);
          large_c -= 1;
        }
      else
        {
          THLongTensor_fastSet1d(larger, large_c-1, large);
          small_c -= 1;
        }
    }

  scalar_t q_min = THTensor_(fastGet1d)(q, inputsize-1);
  scalar_t q_max = q_min;
  scalar_t q_temp;
  for (i=0; i < inputsize; i++)
    {
      q_temp = THTensor_(fastGet1d)(q, i);
      if (q_temp < q_min)
        q_min = q_temp;
      else if (q_temp > q_max)
        q_max = q_temp;
    }
  THArgCheckWithCleanup((q_min > 0),
                        THCleanup(THLongTensor_free(smaller); THLongTensor_free(larger);), 2,
                        "q_min is less than 0");

  if (q_max > 1)
    {
      for (i=0; i < inputsize; i++)
        {
          q_data[i*q->stride(0)] /= q_max;
        }
    }
  for (i=0; i < inputsize; i++)
    {
      // sometimes an large index isn't added to J.
      // fix it by making the probability 1 so that J isn't indexed.
      if(J_data[i] <= 0)
        q_data[i] = 1.0;
    }
  THLongTensor_free(smaller);
  THLongTensor_free(larger);
}
void THTensor_(multinomialAliasDraw)(THLongTensor *self, at::Generator *_generator, THTensor *q, THLongTensor *J, int n_sample)
{
  THArgCheck(q->dim() == 1, 1,
             "expected 1-D probability table, got %d-D probability table instead",
             q->dim());
  THArgCheck(J->dim() == 1, 2,
             "expected 1-D alias table, got %d-D alias table instead",
             J->dim());
  THArgCheck(n_sample > 0, 3, "cannot sample <= 0 samples");
  int64_t K = THLongTensor_nElement(J);
  int64_t i = 0, _mask=0;
  scalar_t _q;
  THLongTensor_resize1d(self, n_sample);
  int64_t rand_ind, sample_idx, J_sample;
  auto gen = at::check_generator_with_default<at::CPUGenerator>(_generator, at::detail::getDefaultCPUGenerator());
  // See Note [Thread-safety and Generators]
  std::lock_guard<std::mutex> lock(gen->mutex_);

  for (i=0; i < n_sample; i++)
    {
      at::uniform_real_distribution<double> uniform(0, K);
      rand_ind = uniform(gen);

      _q = THTensor_(fastGet1d)(q, rand_ind);
      at::bernoulli_distribution<double> bernoulli(_q);
      _mask = static_cast<int64_t>(bernoulli(gen));

      J_sample = THLongTensor_fastGet1d(J, rand_ind);

      sample_idx = J_sample*(1 -_mask) + (rand_ind+1L) * _mask;

      THLongTensor_fastSet1d(self, i, sample_idx-1L);
    }
}
void THTensor_(multinomial)(THLongTensor *self, at::Generator *_generator, THTensor *prob_dist, int n_sample, int with_replacement)
{
  int64_t start_dim = THTensor_(nDimensionLegacyAll)(prob_dist);
  int64_t n_dist;
  int64_t n_categories;
  THDoubleTensor* cum_dist;
  int64_t i,j,k;
  auto gen = at::check_generator_with_default<at::CPUGenerator>(_generator, at::detail::getDefaultCPUGenerator());
  // See Note [Thread-safety and Generators]
  std::lock_guard<std::mutex> lock(gen->mutex_);

  if (start_dim == 1)
  {
    THTensor_(unsqueeze1d)(prob_dist, prob_dist, 0);
  }

  n_dist = THTensor_(size)(prob_dist, 0);
  n_categories = THTensor_(size)(prob_dist, 1);

  THArgCheckWithCleanup(n_sample > 0,
    THCleanup(if (start_dim == 1) THTensor_(squeeze1d)(prob_dist, prob_dist, 0);),
    2,
    "cannot sample n_sample <= 0 samples");

  if (!with_replacement)
  {
    THArgCheckWithCleanup((!with_replacement) && (n_sample <= n_categories),
      THCleanup(if (start_dim == 1) THTensor_(squeeze1d)(prob_dist, prob_dist, 0);),
      2,
      "cannot sample n_sample > prob_dist.size(1) samples without replacement");
  }

  /* cumulative probability distribution vector */
  cum_dist = THDoubleTensor_newWithSize1d(n_categories);

  /* will contain multinomial samples (category indices to be returned) */
  THLongTensor_resize2d(self, n_dist , n_sample);

  auto prod_dist_storage = THTensor_getStoragePtr(prob_dist);
  auto cum_dist_storage = THTensor_getStoragePtr(cum_dist);
  auto self_storage = THTensor_getStoragePtr(self);

  auto prod_dist_offset = prob_dist->storage_offset();
  auto prod_dist_stride_0 = prob_dist->stride(0);
  auto prod_dist_stride_1 = prob_dist->stride(1);

  auto cum_dist_offset = cum_dist->storage_offset();
  auto cum_dist_stride_0 = cum_dist->stride(0);

  auto self_dist_offset = self->storage_offset();
  auto self_dist_stride_0 = self->stride(0);
  auto self_dist_stride_1 = self->stride(1);

  for (i=0; i<n_dist; i++)
  {
    /* Get normalized cumulative distribution from prob distribution */
    double sum = 0;
    double val;
    int n_zeros = 0;
    for (j=0; j<n_categories; j++)
    {
      val = THStorage_(get)( \
        prod_dist_storage, \
        prod_dist_offset+i*prod_dist_stride_0+j*prod_dist_stride_1 \
      );
      THArgCheckWithCleanup((val >= 0),
                            THCleanup(THDoubleTensor_free(cum_dist); if (start_dim == 1) THTensor_(squeeze1d)(prob_dist, prob_dist, 0);),
                            2,
                            "invalid multinomial distribution (encountering probability entry < 0)");
      THArgCheckWithCleanup((std::isfinite(val)),
                            THCleanup(THDoubleTensor_free(cum_dist); if (start_dim == 1) THTensor_(squeeze1d)(prob_dist, prob_dist, 0);),
                            2,
                            "invalid multinomial distribution (encountering probability entry = infinity or NaN)");
      sum += val;
      if (val == 0) {
        n_zeros += 1;
      }
      THDoubleStorage_set(
        cum_dist_storage, \
        cum_dist_offset+j*cum_dist_stride_0, \
        sum \
      );
    }
    THArgCheckWithCleanup((sum > 0),
                          THCleanup(THDoubleTensor_free(cum_dist); if (start_dim == 1) THTensor_(squeeze1d)(prob_dist, prob_dist, 0);),
                          2,
                          "invalid multinomial distribution (sum of probabilities <= 0)");
    THArgCheckWithCleanup((with_replacement || (n_categories - n_zeros >= n_sample)),
                          THCleanup(THDoubleTensor_free(cum_dist); if (start_dim == 1) THTensor_(squeeze1d)(prob_dist, prob_dist, 0);),
                          2,
                          "invalid multinomial distribution (with replacement=False, not enough non-negative category to sample)");
    /* normalize cumulative probability distribution so that last val is 1
    i.e. doesn't assume original prob_dist row sums to one */
    if ( (sum > 0) || ( ( sum < 1.00001) && (sum > 0.99999) ) )
    {
      for (j=0; j<n_categories; j++)
      {
        THDoubleTensor_data(cum_dist)[j*cum_dist_stride_0] /= sum;
      }
    }

    for (j=0; j<n_sample; j++)
    {
      /* sample a probability mass from a uniform distribution */
      at::uniform_real_distribution<double> uniform(0, 1);
      double uniform_sample = uniform(gen);
      /* Do a binary search for the slot in which the prob falls
      ie cum_dist[row][slot-1] < uniform_prob < cum_distr[row][slot] */
      int left_pointer = 0;
      int right_pointer = n_categories;
      int mid_pointer;
      double cum_prob;
      int sample_idx;
      /* Make sure the last cumulative distribution bucket sums to 1 */
      THDoubleTensor_data(cum_dist)[(n_categories-1)*cum_dist_stride_0] = 1;

      while(right_pointer - left_pointer > 0)
      {
          mid_pointer = left_pointer + (right_pointer - left_pointer) / 2;
          cum_prob = THDoubleStorage_get( \
            cum_dist_storage, \
            cum_dist_offset+mid_pointer*cum_dist_stride_0 \
          );
          if (cum_prob < uniform_sample)
          {
            left_pointer = mid_pointer + 1;
          }
          else
          {
            right_pointer = mid_pointer;
          }
      }
      sample_idx = left_pointer;

       /* store in result tensor (will be incremented for lua compat by wrapper) */
      THLongStorage_set( \
        self_storage, \
        self_dist_offset+i*self_dist_stride_0+j*self_dist_stride_1, \
        sample_idx \
      );

      /* Once a sample is drawn, it cannot be drawn again. ie sample without replacement */
      if (!with_replacement && j < n_sample - 1)
      {
        /* update cumulative distribution so that sample cannot be drawn again */
        double diff;
        double new_val = 0;
        double sum;

        if (sample_idx != 0)
        {
          new_val = THDoubleStorage_get( \
            cum_dist_storage, \
            cum_dist_offset+(sample_idx-1)*cum_dist_stride_0 \
          );
        }
        /* marginal cumulative mass (i.e. original probability) of sample */
        diff = THDoubleStorage_get( \
          cum_dist_storage, \
          cum_dist_offset+sample_idx*cum_dist_stride_0 \
        ) - new_val;
        /* new sum of marginals is not one anymore... */
        sum = 1.0 - diff;
        for (k=0; k<n_categories; k++)
        {
          new_val = THDoubleStorage_get( \
            cum_dist_storage, \
            cum_dist_offset+k*cum_dist_stride_0 \
          );
          if (k >= sample_idx)
          {
            /* remove sampled probability mass from later cumulative probabilities */
            new_val -= diff;
          }
          /* make total marginals sum to one */
          new_val /= sum;
          THDoubleStorage_set( \
            cum_dist_storage, \
            cum_dist_offset+k*cum_dist_stride_0, \
            new_val \
          );
        }
      }
    }
  }

  THDoubleTensor_free(cum_dist);

  if (start_dim == 1)
  {
    THLongTensor_resize1d(self, n_sample);
    THTensor_(squeeze1d)(prob_dist, prob_dist, 0);
  }
}
#endif

#if defined(TH_REAL_IS_BYTE)
void THTensor_(getRNGState)(at::Generator *_generator, THTensor *self)
{
  // See Note [Thread-safety and Generators]
  std::lock_guard<std::mutex> lock(_generator->mutex_);
  static const size_t size = sizeof(at::CPUGenerator);
  at::CPUGenerator* rng_state;
  THTensor_(resize1d)(self, size);
  THArgCheck(THTensor_(nElement)(self) == size, 1, "RNG state is wrong size");
  THArgCheck(THTensor_(isContiguous)(self), 1, "RNG state needs to be contiguous");
  rng_state = (at::CPUGenerator*)self->data<scalar_t>();
  auto cast_generator = static_cast<at::CPUGenerator*>(_generator);
  memcpy(rng_state, cast_generator, size);
}

void THTensor_(setRNGState)(at::Generator *_generator, THTensor *self)
{
  // See Note [Thread-safety and Generators]
  std::lock_guard<std::mutex> lock(_generator->mutex_);
  static const size_t size = sizeof(at::CPUGenerator);
  at::CPUGenerator* rng_state;
  THArgCheck(THTensor_(nElement)(self) == size, 1, "RNG state is wrong size");
  THArgCheck(THTensor_(isContiguous)(self), 1, "RNG state needs to be contiguous");
  rng_state = (at::CPUGenerator*)self->data<scalar_t>();
  int64_t is_valid = 0;
  if (_generator->device().type() == rng_state->device().type()) {
    is_valid = 1;
  }
  THArgCheck(is_valid, 1, "Invalid RNG state");
  auto cast_generator = static_cast<at::CPUGenerator*>(_generator);
  memcpy(cast_generator, rng_state, size);
}
#endif
#endif
