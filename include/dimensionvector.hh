/**
 * @file dimensionvector.hh
 * @author Christoph Langenbruch
 * @date 2024-09-27
 *
 */

#ifndef DIMENSIONVECTOR_H
#define DIMENSIONVECTOR_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <limits>
#include <memory>
#include <array>

namespace morefit {

  template<typename T = double> 
  class dimension {
  public:
    std::string name_;
    std::string tex_;
    T min_;
    T max_;
    bool unlimited_;
    dimension(std::string name):
      name_(name),
      tex_(name),
      min_(-std::numeric_limits<T>::infinity()),
      max_(+std::numeric_limits<T>::infinity()),      
      unlimited_(true)
    {
    }
    dimension(std::string name, T min, T max, bool unlimited=false):
      name_(name),
      tex_(name),
      min_(min),
      max_(max),      
      unlimited_(unlimited)
    {
    }
    dimension(std::string name, std::string tex, T min, T max, bool unlimited=false):
      name_(name),
      tex_(tex),
      min_(min),
      max_(max),      
      unlimited_(unlimited)
    {
    }
    //getters
    std::string get_name() const {return name_;}
    std::string get_tex() const {return tex_;}
    std::string get_from_name() const {return "morefit_"+get_name()+"_from";}
    std::string get_to_name() const {return "morefit_"+get_name()+"_to";}
    T get_min() const {return min_;}
    T get_max() const {return max_;}
    bool is_unlimited() const {return unlimited_;}    
    //setters
    void set_name(std::string name) {name_ = name;}
    void set_tex(std::string tex) {tex_ = tex;}
    void set_min(T min) {min_ = min;}
    void set_max(T max) {max_ = max;}
    void set_unlimited(bool unlimited) {unlimited_ = unlimited;}    
  };

  template<typename T = double> 
  class DimensionVector {
  private:
    std::vector<dimension<T>> dimensions_;
  public:
    DimensionVector(std::vector<dimension<T>> dimensions):
      dimensions_(dimensions)
    {
    }
    dimension<T> get_dimension(int idx) const {return dimensions_.at(idx);}
    dimension<T> get_dimension(std::string name) const {
      auto it = find(dimensions_.begin(), dimensions_.end(), [name](dimension<T>& par) { return par.get_name() == name; });
      return *it;
    }
    int size() const {return dimensions_.size();}
    void clear() {dimensions_.clear();}
    void push_back(const dimension<T>& par) {dimensions_.push_back(par);};
    void push_back(const DimensionVector<T>& par_vector) {dimensions_.push_back(par_vector.dimensions_);};    
  };

}

#endif
