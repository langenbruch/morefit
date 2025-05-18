/**
 * @file random.hh
 * @author Christoph Langenbruch
 * @date 2024-11-16
 *
 */

//methods related to random number generation

#ifndef RANDOM_H
#define RANDOM_H

#include <stdint.h>
#ifdef WITH_ROOT
#include "TRandom3.h"
#endif

namespace morefit {

  class RandomGenerator {
  public:
    //return random double in interval from 0 to 1
    virtual double random() = 0;
    //return random float in interval from 0 to 1
    virtual float randomFloat() = 0;
    //return random uint64_t
    virtual uint64_t randomInt64() = 0;
    //set initial seed (derived generators may provide initialisations)
    virtual void setSeed(uint64_t seed) = 0;
    //some generators profit from some burnin for the internal state, depending on seed
    void burnin(unsigned int nevents=10000)
    {
      for (unsigned int i=0; i<nevents; i++)
	random();
    }
    //generate random number distributed according to exponential function
    double exponential(double tau=1.0)
    {
      return -tau*log(random());
    }
    //generate poisson distributed integer, mean mu
    unsigned int poisson(double mean)
    {
      unsigned int result = 0;
      if (mean <= 0.0)
	result = 0;
      else if (mean < 25.0)//chosen same ranges as root here
	{//knuth
	  double p = 1.0;
	  double L = exp(-mean);
	  int k;
	  for (k = 0; p>L; k++)
	    p *= random();
	  result = k-1;
	}
      else if (mean < 1.e9)
	{
	  double sqrt_2mean = sqrt(2.0*mean);
	  double log_mean = log(mean);
	  double g = mean*log_mean - lgamma(mean + 1.0);
	  double r, m, y;	  
	  do {
	    do {
	      y = tan(M_PI*random());
	      m = sqrt_2mean*y + mean;
	    } while (m < 0.0);
	    m = floor(m);
	    r = 0.9*(1.0 + y*y)*exp(m*log_mean - lgamma(m+1.0) - g);
	  } while (random() > r);
	  result = (int)m;
	}
      else//gaussian approximation for very large numbers
	{
	  double r = mean+gaus(0.0, 1.0)*sqrt(mean)+0.5;
	  if (r > std::numeric_limits<unsigned int>::max())
	    {
	      std::cout << "Poisson number exceeds integer limit" << std::endl;
	      assert(0);
	    }
	  result = (int)(r);
	}
      return result;
    }
    //generate random number distributed according to gaussian function
    double gaus(double mean=0.0, double width=1.0)
    {
      //simple polar coordinate method, more optimal algorithms exist
      double u, v, s;
      do {
	u = -1.0+2.0*random();
	v = -1.0+2.0*random();
	s = u*u+v*v;
      }
      while (s >= 1.0);
      double x = u*sqrt(-2.0*log(s)/s);
      return mean + x * width;//wastes one random number
    }
    //generate two random numbers distributed according to gaussian function
    void gaus2(double& a, double& b, double mean=0.0, double width=1.0)
    {
      //simple polar coordinate method, more optimal algorithms exist
      double u, v, s;
      do {
	u = -1.0+2.0*random();
	v = -1.0+2.0*random();
	s = u*u+v*v;
      }
      while (s >= 1.0);
      double x = u*sqrt(-2.0*log(s)/s);
      double y = v*sqrt(-2.0*log(s)/s);
      a = mean + x * width;
      b = mean + y * width;
      return ;
    }
  };

#ifdef WITH_ROOT
  //encapsulate ROOT random number generators
  class Random3: public RandomGenerator {
    friend class TRandom3;
  private:
    //This is the root implementation of the Mersenne Twister
    TRandom3* rnd;
  public:
    Random3(int seed=4357)
    {
      rnd = new TRandom3(seed);
    }
    ~Random3()
    {
      delete rnd;
    }
    virtual void setSeed(uint64_t seed) override
    {
      rnd->SetSeed(seed);
    }
    virtual double random() override
    {
      //note that this just uses 32 bit for the mantissa of a double which is 53 bits
      return rnd->Rndm();
    }
    virtual float randomFloat() override
    {
      return rnd->Rndm();
    }
    virtual uint64_t randomInt64() override
    {
      double a = random();
      double b = random();
      // * 0x1.0p-53
      uint64_t rnda = a / 2.3283064365386963e-10;
      uint64_t rndb = b / 2.3283064365386963e-10;
      return (uint64_t(rnda) << 32) + rndb;      
    }
  };
#endif
  
  //Splitmix64
  //implementation written in 2015 by Sebastiano Vigna (vigna@acm.org)
  class Splitmix64: public RandomGenerator {
  private:
    uint64_t state; 
  public:
    Splitmix64(uint64_t seed): state(seed) {}
    virtual void setSeed(uint64_t seed) override
    {
      state = seed;
    }
    virtual uint64_t randomInt64() override
    {
	uint64_t z = (state += 0x9e3779b97f4a7c15);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
	return z ^ (z >> 31);
    }
    virtual double random() override
    {
      uint64_t x = randomInt64();
      return (x >> 11) * 0x1.0p-53;
    }
    virtual float randomFloat() override
    {
      return random();
    }
  };
  
  //PCG random PCG64DXSM numpy implementation
  //Melissa O'Neill <oneill@pcg-random.org> https://github.com/imneme/pcg-cpp/tree/master
  class PCG64DXSM: public RandomGenerator {
    typedef unsigned __int128 uint128_t;
  private:
    uint128_t state;
    uint128_t inc;
  public:
    PCG64DXSM(uint128_t seed, uint128_t seed_inc=1442695040888963407ULL): state(seed), inc(seed_inc) {}
    void setSeed(uint128_t seed)
    {
      state = seed;
      return;
    }
    virtual void setSeed(uint64_t seed) override
    {
      state = seed;
      return;
    }
    void setSeed(uint64_t seed, uint128_t seed_inc=1442695040888963407ULL)
    {
      inc = seed_inc;
      Splitmix64 s(seed);
      state = (uint128_t(s.randomInt64())<<64) + s.randomInt64();
      return;
    }
    virtual uint64_t randomInt64() override
    {
      // cheap (half-width) multiplier
      const uint64_t mul = 15750249268501108917ULL;//0x2360ed051fc65da44385df649fccf645
      // linear congruential generator 
      state = state * mul + inc;
      // DXSM (double xor shift multiply) permuted output 
      uint64_t hi = (uint64_t)(state >> 64);
      uint64_t lo = (uint64_t)(state | 1);
      hi ^= hi >> 32;
      hi *= mul;
      hi ^= hi >> 48;
      hi *= lo;
      return hi;
    }
    virtual double random() override
    {
      uint64_t x = randomInt64();
      return (x >> 11) * 0x1.0p-53;
    }
    virtual float randomFloat() override
    {
      return random();
    }
  };
      
  //Xoshiro256++ Written in 2019 by David Blackman and Sebastiano Vigna (vigna@acm.org)
  //https://prng.di.unimi.it/xoshiro256plusplus.c
  class Xoshiro256pp: public RandomGenerator {
  private:
    uint64_t state[4];
    inline uint64_t rol64(uint64_t x, int k)
    {
      return (x << k) | (x >> (64 - k));
    }
  public:
    Xoshiro256pp(uint64_t seed[4])
    {
      state[0] = seed[0];
      state[1] = seed[1];
      state[2] = seed[2];
      state[3] = seed[3];
    }
    Xoshiro256pp(uint64_t seed = 3457)
    {
      Splitmix64 s(seed);
      state[0] = s.randomInt64();
      state[1] = s.randomInt64();
      state[2] = s.randomInt64();
      state[3] = s.randomInt64();
    }
    void setSeed(uint64_t seed[4])
    {
      state[0] = seed[0];
      state[1] = seed[1];
      state[2] = seed[2];
      state[3] = seed[3];
    }
    uint64_t getSeed(int idx) const
    {
      return state[idx];
    }
    virtual void setSeed(uint64_t seed) override
    {
      Splitmix64 s(seed);
      state[0] = s.randomInt64();
      state[1] = s.randomInt64();
      state[2] = s.randomInt64();
      state[3] = s.randomInt64();      
    }    
    virtual uint64_t randomInt64() override
    {
      uint64_t const result = rol64(state[0] + state[3], 23) + state[0];
      uint64_t const t = state[1] << 17;
      
      state[2] ^= state[0];
      state[3] ^= state[1];
      state[1] ^= state[2];
      state[0] ^= state[3];
      
      state[2] ^= t;
      state[3] = rol64(state[3], 45);
      
      return result;
    }
    virtual double random() override
    {
      uint64_t x = randomInt64();
      return (x >> 11) * 0x1.0p-53;
    }
    virtual float randomFloat() override
    {
      return random();
    }
    void jump()
    {
	static const uint64_t JUMP[4] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };
	uint64_t s0 = 0;
	uint64_t s1 = 0;
	uint64_t s2 = 0;
	uint64_t s3 = 0;
	for(int i = 0; i < 4; i++)
		for(int b = 0; b < 64; b++) {
			if (JUMP[i] & UINT64_C(1) << b) {
				s0 ^= state[0];
				s1 ^= state[1];
				s2 ^= state[2];
				s3 ^= state[3];
			}
			randomInt64();
		}
	state[0] = s0;
	state[1] = s1;
	state[2] = s2;
	state[3] = s3;
    }
  };

  //Xoshiro128++ Written in 2019 by David Blackman and Sebastiano Vigna (vigna@acm.org)
  //https://prng.di.unimi.it/xoshiro128plusplus.c
  class Xoshiro128pp: public RandomGenerator {
  private:
    uint32_t state[4];
    static inline uint32_t rol32(uint32_t x, int k)
    {
      return (x << k) | (x >> (32 - k));
    }
  public:
    Xoshiro128pp(uint32_t seed[4])
    {
      state[0] = seed[0];
      state[1] = seed[1];
      state[2] = seed[2];
      state[3] = seed[3];
    }
    Xoshiro128pp(uint32_t seed = 3457)
    {
      Splitmix64 s(seed);
      state[0] = s.randomInt64() >> 32;
      state[1] = s.randomInt64() >> 32;
      state[2] = s.randomInt64() >> 32;
      state[3] = s.randomInt64() >> 32;
    }
    void setSeed(uint32_t seed[4])
    {
      state[0] = seed[0];
      state[1] = seed[1];
      state[2] = seed[2];
      state[3] = seed[3];
    }
    uint32_t getSeed(int idx) const
    {
      return state[idx];
    }
    virtual void setSeed(uint64_t seed) override
    {
      Splitmix64 s(seed);
      state[0] = s.randomInt64() >> 32;
      state[1] = s.randomInt64() >> 32;
      state[2] = s.randomInt64() >> 32;
      state[3] = s.randomInt64() >> 32;      
    }     
    virtual uint64_t randomInt64() override
    {
      return (uint64_t(randomInt32())<<32) + randomInt32();
    }
    inline uint32_t randomInt32()
    {
      uint32_t const result = rol32(state[0] + state[3], 7) + state[0];
      uint32_t const t = state[1] << 9;
      
      state[2] ^= state[0];
      state[3] ^= state[1];
      state[1] ^= state[2];
      state[0] ^= state[3];
      
      state[2] ^= t;
      state[3] = rol32(state[3], 11);
      
      return result;
    }
    virtual double random() override
    {
      return  (float) (randomInt32() * 2.3283064365386963e-10);
    }
    virtual float randomFloat() override
    {
      return  (float) (randomInt32() * 2.3283064365386963e-10);
    }
    void jump()
    {     
	static const uint32_t JUMP[4] = { 0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b };
	uint32_t s0 = 0;
	uint32_t s1 = 0;
	uint32_t s2 = 0;
	uint32_t s3 = 0;
	
	for(int i = 0; i < 4; i++)
		for(int b = 0; b < 32; b++) {
			if (JUMP[i] & UINT32_C(1) << b) {
				s0 ^= state[0];
				s1 ^= state[1];
				s2 ^= state[2];
				s3 ^= state[3];
			}
			randomInt32();
		}
	
	/*
	for(int b = 0; b < 32; b++) {
	  if (JUMP[0] & UINT32_C(1) << b) {
	    s0 ^= state[0];
	    s1 ^= state[1];
	    s2 ^= state[2];
	    s3 ^= state[3];
	  }
	  randomInt32();
	}
	for(int b = 0; b < 32; b++) {
	  if (JUMP[1] & UINT32_C(1) << b) {
	    s0 ^= state[0];
	    s1 ^= state[1];
	    s2 ^= state[2];
	    s3 ^= state[3];
	  }
	  randomInt32();
	}
	for(int b = 0; b < 32; b++) {
	  if (JUMP[2] & UINT32_C(1) << b) {
	    s0 ^= state[0];
	    s1 ^= state[1];
	    s2 ^= state[2];
	    s3 ^= state[3];
	  }
	  randomInt32();
	}
	for(int b = 0; b < 32; b++) {
	  if (JUMP[3] & UINT32_C(1) << b) {
	    s0 ^= state[0];
	    s1 ^= state[1];
	    s2 ^= state[2];
	    s3 ^= state[3];
	  }
	  randomInt32();
	}
	*/
	
	state[0] = s0;
	state[1] = s1;
	state[2] = s2;
	state[3] = s3;
    }
  };

  //Mersenne twister implementation following
  //Matsumoto and T. Nishimura, Mersenne Twister: A 623-diminsionally equidistributed uniform pseudorandom number generator ACM Transactions on Modeling and Computer Simulation, Vol. 8, No. 1, January 1998, pp 3–30.
  class MersenneTwister: public RandomGenerator {
  private:
    //constants
    static constexpr int32_t n = 624;
    static constexpr int32_t m = 397;
    static constexpr int32_t w = 32;
    static constexpr int32_t r = 31;
    static constexpr uint64_t UPPER_MASK = (0xffffffffUL << r);
    static constexpr uint64_t LOWER_MASK = (0xffffffffUL >> (w-r));
    static constexpr uint64_t MATRIX_A = 0x9908b0dfUL;
    //internal state
    uint32_t state_array[n];
    int state_index;
    void initState(uint32_t seed)
    {
      state_array[0] = seed;
      for (int i=1; i<n; i++) {
	//See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier
        state_array[i] = (1812433253UL * (state_array[i-1] ^ (state_array[i-1] >> 30)) + i); 
	//for 32 bit machines
        state_array[i] &= 0xffffffffUL;
      }
      state_index = 624;
    }
    public:
    MersenneTwister(uint32_t seed = 3457)
    {
      initState(seed);
    }
    virtual void setSeed(uint64_t seed) override
    {
      initState(seed);
    }
    virtual double random() override
    {
      return randomRoot();//might want to actually use randomDouble as default instead
    }
    //this should use two 32bit random numbers (strictly speaking necessary for 53bit mantissa?)
    double randomDouble()
    {
      uint64_t a = randomInt32()>>5, b = randomInt32()>>6; 
      return (a*67108864.0+b)*(1.0/9007199254740992.0); 
    }
    //random number generator corresponding to root implementation, note this does not use full 53 bit for double mantissa
    double randomRoot()
    {
      uint32_t y = randomInt32();
      if (y) //this will only use 32 bit
	return ( (double) y * 2.3283064365386963e-10);
      return random();      
    }
    virtual float randomFloat() override
    {
      uint32_t y = randomInt32();
      if (y) 
	return ( (float) y * 2.3283064365386963e-10);
      return random();      
    }
    //generates a random uint32 number on [0,0xffffffff]-interval 
    inline uint32_t randomInt32()
    {
      uint32_t y;
      if (state_index >= n) // generate N words at one time 
	{
	  int kk;	  
	  for (kk=0; kk<n-m; kk++)
	    {
	      y = (state_array[kk] & UPPER_MASK) | (state_array[kk+1] & LOWER_MASK);
	      state_array[kk] = state_array[kk+m] ^ (y >> 1) ^ ((y&0x1) ? MATRIX_A : 0x0UL);
	    }
	  for (; kk<n-1; kk++)
	    {
	      y = (state_array[kk] & UPPER_MASK) | (state_array[kk+1] & LOWER_MASK);
	      state_array[kk] = state_array[kk+(m-n)] ^ (y >> 1) ^ ((y&0x1) ? MATRIX_A : 0x0UL);
	    }
	  y = (state_array[n-1] & UPPER_MASK) | (state_array[0] & LOWER_MASK);
	  state_array[n-1] = state_array[m-1] ^ (y >> 1) ^ ((y&0x1) ? MATRIX_A : 0x0UL);
	  
	  state_index = 0;
	}      
      y = state_array[state_index++];      
      // Tempering 
      y ^= (y >> 11);
      y ^= (y << 7) & 0x9d2c5680UL;
      y ^= (y << 15) & 0xefc60000UL;
      y ^= (y >> 18);      
      return y;
    }
    virtual uint64_t randomInt64() override
    {
      return (uint64_t(randomInt32()) << 32) + randomInt32();
    }
  };

  
}

#endif
