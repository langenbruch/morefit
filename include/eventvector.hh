/**
 * @file eventvector.hh
 * @author Christoph Langenbruch
 * @date 2023-02-20
 *
 */

#ifndef EVENTVECTOR_H
#define EVENTVECTOR_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <limits>
#include <memory>
#include <array>

#ifdef WITH_ROOT
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TLeaf.h"
#endif

#include "dimensionvector.hh"

namespace morefit {
  
  template<typename kernelT, typename evalT=double> 
  class EventVector {//SoA
  private:
    unsigned int nevents_;
    std::vector<dimension<evalT>*> dimensions_; 
    unsigned int padding_;
    bool padded_;
    int weight_idx_;
    kernelT* data_;
  public:
    EventVector()://empty event vector
      nevents_(0),
      padding_(false),
      padded_(false),
      weight_idx_(-1),
      data_(nullptr)
    {
    }
    EventVector(std::vector<dimension<evalT>*> dimensions, 
		unsigned int nevents=0, bool padded=false, int padding=0):
      nevents_(nevents),
      dimensions_(dimensions),
      padding_(padding),
      padded_(padded),
      weight_idx_(-1),
      data_(dimensions.size()*nevents > 0 ? new kernelT[dimensions.size()*(padded ? nevents_padded(nevents) : nevents)] : nullptr)
    {
    }
    EventVector(const EventVector<kernelT, evalT>& rhs):
      nevents_(rhs.nevents_),
      dimensions_(rhs.dimensions_),
      padding_(rhs.padding_),
      padded_(rhs.padded_),
      weight_idx_(-1)
    {
      if (rhs.dimensions_.size()*rhs.nevents_ > 0)
	{
	  data_ = new kernelT[rhs.dimensions_.size()*(rhs.padded_ ? rhs.nevents_padded(rhs.nevents_) : rhs.nevents_)];
	  std::copy(rhs.data_, rhs.data_ + rhs.dimensions_.size()*(rhs.padded_ ? rhs.nevents_padded(rhs.nevents_) : rhs.nevents_), data_);
	}
    }
    ~EventVector()
    {
      if (data_)
	delete[] data_;
    };
    int set_event_weight(std::string weight_name="")
    {
      int idx = -1;
      for (unsigned int i=0; i<dimensions_.size(); i++)
	if (dimensions_.at(i)->get_name() == weight_name)
	{
	  idx = i;
	  break;
	}
      if (weight_name != "" && idx == -1)
	{
	  std::cout << "Could not find event weight variable " << weight_name << " in vector." << std::endl;
	  assert(0);
	}
      weight_idx_ = idx;
      return weight_idx_;
    }
    int event_weight_idx() const
    {
      return weight_idx_;
    }
    std::string event_weight_name() const
    {
      if (weight_idx_ != -1)
	return dimensions_.at(weight_idx_)->get_name();
      else
	return "";
    }
    bool has_event_weight() const
    {
      return (weight_idx_ >= 0);
    }
    inline unsigned int nevents_padded(unsigned int nevents) const
    {
      return nevents + ((padding_ - (nevents % padding_)) % padding_);
    }
    unsigned int nevents_padded() const
    {
      if (padded_)
	return nevents_ + ((padding_ - (nevents_ % padding_)) % padding_);
      else
	return nevents_;
    }
    unsigned int ndimensions() const
    {
      return dimensions_.size();
    };
    void set_padding(bool padded, unsigned int new_padding)
    {
      if (padded_ != padded || padding_ != new_padding)
	{
	  int size_before = buffer_size();
	  unsigned int padded_nevents_old = padded_ ? nevents_padded(nevents_) : nevents_;
	  unsigned int padded_nevents_new = padded ? nevents_ + ((new_padding - (nevents_ % new_padding)) % new_padding) : nevents_;
	  kernelT* new_data = nullptr;
	  if (dimensions_.size()*padded_nevents_new > 0)
	    new_data = new kernelT[dimensions_.size()*padded_nevents_new];//nb. entries may be uninitialised      
	  unsigned int nevents_to_copy = padded_nevents_new > padded_nevents_old ?  padded_nevents_old : padded_nevents_new;
	  for (unsigned int i=0; i<dimensions_.size(); i++)
	    std::copy(data_+i*padded_nevents_old, data_+i*padded_nevents_old+nevents_to_copy, new_data+i*padded_nevents_new);
	  int size_after = dimensions_.size()*padded_nevents_new*sizeof(kernelT);
	  std::cout << "Event vector padding changed from " << (padded_ ? std::to_string(padding_) : "NOT PADDED") << " to " << (padded ? std::to_string(new_padding) : "NOT PADDED")
		    << " resulting in buffer size change from " << size_before << " to " << size_after << std::endl;
	  if (data_)
	    delete[] data_;
	  if (new_data)
	    data_ = new_data;
	  padded_ = padded;
	  padding_ = new_padding;
	}
    }
    bool is_padded() const
    {
      return padded_;
    }
    unsigned int padding() const
    {
      return padding_;
    }
    
    std::vector<dimension<evalT>*> get_dimensions() const
    {
      return dimensions_;
    }    
    std::vector<dimension<evalT>> copy_dimensions() const
    {
      std::vector<dimension<evalT>> result;
      for (unsigned int i=0; i<dimensions_.size(); i++)
	result.push_back(*dimensions_.at(i));
      return result;
    }    
    std::vector<std::string> get_dimensions_str() const
    {
      std::vector<std::string> results;
      for (const auto& dim : dimensions_)
	results.push_back(dim->get_name());
      return results;
    }    
    unsigned int size_t() const
    {
      return sizeof(kernelT);
    };
    const unsigned int& nevents() const
    {
      return nevents_;
    };
    //resize and copy existing events
    void resize(unsigned int nevents)//this operation is more expensive for SoA
    {
      if (nevents != nevents_)
	{
	  unsigned int padded_nevents_new = padded_ ? nevents_padded(nevents) : nevents;
	  unsigned int padded_nevents_old = padded_ ? nevents_padded(nevents_) : nevents_;
	  kernelT* new_data = nullptr;
	  if (dimensions_.size()*padded_nevents_new > 0)
	    new_data = new kernelT[dimensions_.size()*padded_nevents_new];//nb. entries may be uninitialised
	  unsigned int nevents_to_copy = padded_nevents_new > padded_nevents_old ?  padded_nevents_old : padded_nevents_new;
	  for (unsigned int i=0; i<dimensions_.size(); i++)
	    std::copy(data_+i*padded_nevents_old, data_+i*padded_nevents_old+nevents_to_copy, new_data+i*padded_nevents_new);
	  if (data_)
	    delete[] data_;
	  if (new_data)
	    data_ = new_data;
	  nevents_ = nevents;
	}
    }
    //interleave additional dimensions to vector and copy existing data
    void add_dimensions(std::vector<dimension<evalT>*> new_dimensions)//this operation is less expensive for SoA
    {
      unsigned int padded_nevents = padded_ ? nevents_padded(nevents_) : nevents_;
      std::vector<dimension<evalT>*> all_dimensions(dimensions_);
      all_dimensions.insert(all_dimensions.end(), new_dimensions.begin(), new_dimensions.end());
      kernelT* new_data = nullptr;
      if (all_dimensions.size()*padded_nevents > 0)
	new_data = new kernelT[all_dimensions.size()*padded_nevents];//nb. values for new dimension not initialised
      std::copy(data_, data_+dimensions_.size()*padded_nevents, new_data);
      if (data_)
	delete[] data_;
      if (new_data)
	data_ = new_data;
      dimensions_ = all_dimensions;      
    }
    //return buffer size in bytes
    unsigned int buffer_size() const
    {
      return (padded_ ? (nevents_padded(nevents())) : nevents())*ndimensions()*size_t();
    }
    //return index of dimension "d"
    int dim_index(std::string d) const
    {
      for (unsigned int i=0; i<dimensions_.size(); i++)
	if (dimensions_.at(i)->get_name() == d)
	  return i;
      return -1;
    }
    kernelT* get_data() const
    {
      return data_;
    }
    const kernelT& operator()(unsigned int n, unsigned int d) const
    {
      //n is event, d is dimension, d*npadded_events+n
      unsigned int padded_nevents = padded_ ? (nevents_padded(nevents_)) : nevents_;
      return data_[d*padded_nevents+n];
    }
    const kernelT& operator()(unsigned int n, std::string d) const
    {
      int idx = dim_index(d);
      assert(idx != -1);
      unsigned int padded_nevents = padded_ ? (nevents_padded(nevents_)) : nevents_;
      return data_[idx*padded_nevents+n];
    }
    kernelT& operator()(unsigned int n, unsigned int d)
    {
      unsigned int padded_nevents = padded_ ? (nevents_padded(nevents_)) : nevents_;
      return data_[d*padded_nevents+n];
    }
    kernelT& operator()(unsigned int n, std::string d)
    {
      int idx = dim_index(d);
      assert(idx != -1);
      unsigned int padded_nevents = padded_ ? (nevents_padded(nevents_)) : nevents_;
      return data_[idx*padded_nevents+n];
    }
    //pretty print vector to stdout
    void print(int precision=6, int width=10, int max_events=-1, bool print_header=true, bool print_info=true, bool print_index=true) {
      if (print_info)
	std::cout << "morefit::EventVector with " << ndimensions() << " dimensions consisting of " << nevents() << " events resulting in " << buffer_size() << " bytes total buffer size" << std::endl;
      if (print_header)
	{
	  if (print_index)
	    std::cout << std::setw(width) << "index";
	  for (auto it = dimensions_.begin(); it != dimensions_.end(); it++)
	    std::cout << std::setw(width) << (*it)->get_name();

	  std::cout << std::endl;
	}
      if (max_events > -1)
	max_events = (max_events > int(nevents()) ? nevents() : max_events);
      else
	max_events = nevents();
      for (int i=0; i<max_events; i++)
	{
	  if (print_index)
	    std::cout << std::setw(width) << i;
	  for (unsigned int j=0; j<ndimensions(); j++)
	    std::cout << std::setw(width) << std::setprecision(precision) << operator()(i,j);	  
	  std::cout << std::endl;
	}
      if (max_events < int(nevents()))
	std::cout << "... and " << nevents()-max_events << " events more." << std::endl;
      std::cout << (padded_ ? ("morefit::EventVector is padded to "+std::to_string(padding_)) : "morefit::EventVector is not padded")
		<< " resulting in an overhead of " << (padded_ ? (nevents_padded(nevents()) - nevents()) : 0) << " events per dimension "
		<< "for a total overhead of " << (padded_ ? (nevents_padded(nevents()) - nevents())*ndimensions()*size_t() : 0)
		<< "=" << (padded_ ? (nevents_padded(nevents()) - nevents()) : 0) << "*" << ndimensions() << "*" << size_t() << " bytes" 
		<< std::endl;
    }
  };//EventVector

  //load EventVector from root TTree, 
#ifdef WITH_ROOT
  template <typename kernelT=double, typename evalT=double>
  EventVector<kernelT, evalT> load_event_vector(std::string filename, std::string treename, std::vector<std::string> dimensions, std::vector<std::string> branchnames={}, int max_event=-1)
  {
    //assume branchnames have same names as dimensions
    if (branchnames.size() == 0)
      branchnames = std::vector<std::string>(dimensions);
    assert(dimensions.size() == branchnames.size());
    //TODO safeties
    TFile* f = new TFile(filename.c_str(), "READ");
    TTree* t = dynamic_cast<TTree*>(f->Get(treename.c_str()));
    unsigned int nevents = t->GetEntries();
    if (max_event > 0 && max_event < int(nevents))
      nevents = max_event;
    EventVector<kernelT, evalT> result(dimensions, nevents);
    TBranch* branches[dimensions.size()];
    TLeaf* leaves[dimensions.size()];    
    t->SetBranchStatus("*", 0);
    for (unsigned int i=0; i<dimensions.size(); i++)
      {
	t->SetBranchStatus(branchnames.at(i).c_str(), 1);
	branches[i] = t->GetBranch(branchnames.at(i).c_str());
	leaves[i] = t->GetLeaf(branchnames.at(i).c_str());
      }
    for (unsigned int i=0; i<dimensions.size(); i++)
      assert(branches[i]);
    for (unsigned int i=0; i<nevents; i++)
      {
	t->GetEntry(i);
	for (unsigned int j=0; j<dimensions.size(); j++)
	  result(i,j) = leaves[j]->GetValue();
      }
    f->Close();
    delete f;
    return result;
  }
#endif

};

#endif
