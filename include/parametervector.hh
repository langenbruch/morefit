/**
 * @file parametervector.hh
 * @author Christoph Langenbruch
 * @date 2023-02-20
 *
 */

#ifndef PARAMETERVECTOR_H
#define PARAMETERVECTOR_H

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
  class parameter {
  public:
    std::string name_;
    std::string tex_;
    T value_;
    T min_;
    T max_;
    T step_size_;
    T start_value_;
    T error_;
    T error_up_;
    T error_down_;
    bool unlimited_;
    int minuit_idx_;
    parameter(std::string name, T value):
      name_(name),
      tex_(name),
      value_(value),
      min_(-std::numeric_limits<T>::infinity),
      max_(+std::numeric_limits<T>::infinity),      
      step_size_(0.0),
      start_value_(value),
      error_(0.0),
      error_up_(0.0),
      error_down_(0.0),
      unlimited_(true),
      minuit_idx_(-1)
    {
    }
    parameter(std::string name, T value, T min, T max, T step_size, bool unlimited=false):
      name_(name),
      tex_(name),
      value_(value),
      min_(min),
      max_(max),      
      step_size_(step_size),
      start_value_(value),
      error_(0.0),
      error_up_(0.0),
      error_down_(0.0),
      unlimited_(unlimited),
      minuit_idx_(-1)
    {
    }
    parameter(std::string name, std::string tex, T value, T min, T max, T step_size, bool unlimited=false):
      name_(name),
      tex_(tex),
      value_(value),
      min_(min),
      max_(max),      
      step_size_(step_size),
      start_value_(value),
      error_(0.0),
      error_up_(0.0),
      error_down_(0.0),
      unlimited_(unlimited),
      minuit_idx_(-1)
    {
    }
    //re-initialisations
    void init(std::string name, T value)
    {
      name_ = name;
      tex_ = name;
      value_ = value;
      min_ = -std::numeric_limits<T>::infinity;
      max_ = +std::numeric_limits<T>::infinity;      
      step_size_ = 0.0;
      start_value_ = value;
      error_ = 0.0;
      error_up_ = 0.0;
      error_down_ = 0.0;
      unlimited_ = true;
      minuit_idx_ = -1;
    }
    void init(std::string name, T value, T min, T max, T step_size, bool unlimited=false)
    {
      name_ = name;
      tex_ = name;
      value_ = value;
      min_ = min;
      max_ = max;      
      step_size_ = step_size;
      start_value_ = value;
      error_ = 0.0;
      error_up_ = 0.0;
      error_down_ = 0.0;
      unlimited_ = unlimited;
      minuit_idx_ = -1;
    }
    void init(std::string name, std::string tex, T value, T min, T max, T step_size, bool unlimited=false)
    {
      name_ = name;
      tex_ = tex;
      value_ = value;
      min_ = min;
      max_ = max;      
      step_size_ = step_size;
      start_value_ = value;
      error_ = 0.0;
      error_up_ = 0.0;
      error_down_ = 0.0;
      unlimited_ = unlimited;
      minuit_idx_ = -1;
    }
    //getters
    std::string get_name() const {return name_;}
    std::string get_tex() const {return tex_;}
    T get_value() const {return value_;}
    T get_min() const {return min_;}
    T get_max() const {return max_;}
    T get_step_size() const {return step_size_;}
    T get_start_value() const {return start_value_;}
    T get_error() const {return error_;}
    T get_error_up() const {return error_up_;}
    T get_error_down() const {return error_down_;}
    bool is_unlimited() const {return unlimited_;}    
    bool is_constant() const {return (get_step_size() == 0.0);}
    int get_minuit_idx() const {return minuit_idx_;}
    //setters
    void set_name(std::string name) {name_ = name;}
    void set_tex(std::string tex) {tex_ = tex;}
    void set_value(T value) {value_  = value;}
    void set_min(T min) {min_ = min;}
    void set_max(T max) {max_ = max;}
    void set_step_size(T step_size) {step_size_ = step_size;}
    void set_start_value(T start_value) {start_value_ = start_value;}
    void set_error(T error) {error_ = error;}
    void set_error_up(T error_up) {error_up_ = error_up;}
    void set_error_down(T error_down) {error_down_ = error_down;}
    void set_unlimited(bool unlimited) {unlimited_ = unlimited;}    
    void set_constant() {step_size_ = 0.0;}   
    void set_minuit_idx(int minuit_idx) {minuit_idx_ = minuit_idx;}
  };

  template<typename T = double> 
  class ParameterVector {
  private:
    std::vector<parameter<T>> parameters_;
  public:
    ParameterVector(std::vector<parameter<T>> parameters):
      parameters_(parameters)
    {
    }
    parameter<T> get_parameter(int idx) const {return parameters_.at(idx);}
    parameter<T> get_parameter(std::string name) const {
      auto it = find(parameters_.begin(), parameters_.end(), [name](parameter<T>& par) { return par.get_name() == name; });
      return *it;
    }
    int size() const {return parameters_.size();}
    void clear() {parameters_.clear();}
    void push_back(const parameter<T>& par) {parameters_.push_back(par);};
    void push_back(const ParameterVector<T>& par_vector) {parameters_.push_back(par_vector.parameters_);};    
  };

}

#endif
