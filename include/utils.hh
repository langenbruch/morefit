/**
 * @file utils.hh
 * @author Christoph Langenbruch
 * @date 2024-11-26
 *
 */

#ifndef UTILS_H
#define UTILS_H

namespace morefit {
  
  //  __attribute__((optimize("no-fast-math")))
  template<typename returnT=double, typename vectorT=double>
  returnT kahan_summation(const std::vector<vectorT>& values)
  {
    returnT sum = 0.0;
    returnT c = 0.0;
    for (vectorT value : values) {
      returnT y = value - c;
      volatile returnT t = sum + y;
      c = (t - sum) - y;
      sum = t;
    }
    return sum;
  }

  template<typename returnT=double, typename vectorT=double>
  returnT kahan_summation(const EventVector<vectorT, returnT>& data, int dim=0)
  {
    returnT sum = 0.0;
    returnT c = 0.0;
    for (unsigned int i=0; i<data.nevents(); i++) {
      vectorT value = data(i, dim);
      returnT y = value - c;
      volatile returnT t = sum + y;
      c = (t - sum) - y;
      sum = t;
    }
    return sum;
  }
  
  //returns true if all elements in vector are unique
  bool elements_unique(const std::vector<std::string>& v)
  {
    std::vector<std::string> tmp(v);
    std::sort(tmp.begin(), tmp.end());
    auto last = std::unique(tmp.begin(), tmp.end());
    if (last == tmp.end())
      return true;
    else
      return false;
  }
  
  template<typename returnT=double, typename vectorT=double>
  void mean_rms(const std::vector<vectorT>& values, returnT& mean, returnT& rms)
  {
    returnT sum = 0.0;
    returnT c = 0.0;
    for (vectorT value : values) {
      returnT y = value - c;
      volatile returnT t = sum + y;
      c = (t - sum) - y;
      sum = t;
    }
    mean = sum/values.size();
    rms = 0.0;
    for (vectorT value : values) 
      rms += (value - mean)*(value - mean);
    rms /= values.size();
    rms = sqrt(rms);
    return ;
  }
  
}

#endif
