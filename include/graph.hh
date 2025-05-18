/**
 * @file computegraph.hh
 * @author Christoph Langenbruch
 * @date 2023-02-20
 *
 */

#ifndef COMPUTEGRAPH_H
#define COMPUTEGRAPH_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <limits>
#include <memory>
#include <array>
#include <math.h>
#include <fstream>

#include "parametervector.hh"
#include "morefit.hh"

namespace morefit {

  //variadic templates to allow for construction of a vector of std::unique_ptr via make_vector<std::unique_ptr<class>>
  template<typename T>
  void multi_emplace(std::vector<T> &vec)
  {
  }
  
  template<typename T, typename T1, typename... Types>
  void multi_emplace(std::vector<T> &vec, T1&& t1, Types&&... args)
  {
    vec.emplace_back(std::move(t1));
    multi_emplace(vec, args...);
  }
  
  template<typename T, typename... Types>
  std::vector<T> make_vector(Types&&... args)
  {
    std::vector<T> result;
    multi_emplace(result, args...);
    return result;
  }
  
  template<typename T, typename... Types>
  std::vector<std::unique_ptr<T>> make_unique_vector(Types&&... args)
  {
    std::vector<std::unique_ptr<T>> result;
    multi_emplace(result, args...);
    return result;
  }

  template <typename kernelT, typename evalT> 
  class ComputeGraphNode;
  template <typename kernelT, typename evalT> 
  class SumNode;
  template <typename kernelT, typename evalT> 
  class ProdNode;
  template <typename kernelT, typename evalT> 
  class ConstantNode;  
  template <typename kernelT, typename evalT> 
  class VariableNode;  
  template <typename kernelT, typename evalT> 
  class ExpNode;
  template <typename kernelT, typename evalT> 
  class InvNode;
  template <typename kernelT, typename evalT> 
  class LogNode;
  template <typename kernelT, typename evalT> 
  class PowNode;
  template <typename kernelT, typename evalT> 
  class NegNode;
  template <typename kernelT, typename evalT> 
  class SqrtNode;
  template <typename kernelT, typename evalT> 
  class SinNode;
  template <typename kernelT, typename evalT> 
  class AsinNode;
  template <typename kernelT, typename evalT> 
  class CosNode;
  template <typename kernelT, typename evalT> 
  class AcosNode;
  template <typename kernelT, typename evalT> 
  class TanNode;
  template <typename kernelT, typename evalT> 
  class AtanNode;
  template <typename kernelT, typename evalT> 
  class ErfNode;

  template<typename kernelT, typename evalT>
  class ComputeGraphNode {
  public:
    std::vector< std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> children_;
    ComputeGraphNode(std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> children)
      : children_(std::move(children))
    {
    }
    ComputeGraphNode(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> child)
    {
      children_.emplace_back(std::move(child));
    }
    ComputeGraphNode()
    {
    }
    virtual ~ComputeGraphNode() = default;    
    //make a copy of this compute graph including all its children
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> copy() const = 0;
    //make a copy of all children
    std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> copy_children() const
    {
      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> result;
      for (unsigned int i=0; i<children_.size(); i++)
	result.emplace_back(std::move(children_.at(i)->copy()));
      return result;
    }
    //renames single variable, does change this expression (ie. is non-const)
    virtual void rename_variable(const std::string& variable, const std::string& new_name)
    {
      for (unsigned int i=0; i<children_.size(); i++)
	children_.at(i)->rename_variable(variable, new_name);
    }
    //get string used in kernel
    virtual std::string get_kernel() const = 0;
    //pretty-print compute graph to stdout
    virtual void print(unsigned int level = 0) const = 0;
    //derivative to variable
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> diff(std::string variable) const = 0;
    //substitute named variables (can be dimensions/parameters) with constants    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> substitute(const std::vector<std::string>& variables, const std::vector<double>& values) const = 0;    
    //perform rudimentary simplifications
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> simplify() const = 0;
    //approximate cost of evaluation
    virtual double cost() const = 0;
    //type as string to be used in kernel
    std::string kernelT_str() const
    {
      if constexpr (std::is_same_v<float, kernelT>) //evaluated at compile time
	return "float";
      else if constexpr (std::is_same_v<double, kernelT>)
	return "double";
      else if constexpr (std::is_same_v<long double, kernelT>)
	return "long double";
      else
	{
	  std::cout << "Unknown kernel type" << std::endl;
	  assert(0);
	}
    }
    std::string evalT_str() const
    {
      if constexpr (std::is_same_v<float, evalT>) //evaluated at compile time
	return "float";
      else if constexpr (std::is_same_v<double, evalT>)
	return "double";
      else if constexpr (std::is_same_v<long double, evalT>)
	return "long double";
      else
	{
	  std::cout << "Unknown eval type" << std::endl;
	  assert(0);
	}
    }
    //check if tree contains given variable, simplifies differentiation
    virtual bool variable_in_tree(const std::string& variable) const
    {
      bool result = false;
      for (unsigned int i=0; i<children_.size(); i++)
	if (children_.at(i)->variable_in_tree(variable))
	  {
	    result = true;
	    break;
	  }
      return result;
    }
    //check if tree contains any other variables (used to determine if subtree can be buffered, eg if only depends on parameters)
    virtual bool contains_other_variables(const std::vector<std::string>& variables) const
    {
      bool result = false;
      for (unsigned int i=0; i<children_.size(); i++)
	if (children_.at(i)->contains_other_variables(variables))
	  {
	    result = true;
	    break;
	  }
      return result;      
    }
    //buffer constant terms, returns computegraph replacing bufferable terms with variables, arguments are references to vectors where these terms are stored
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> optimize_buffering_constant_terms(std::vector<std::string>& buffernames, std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>& bufferexpressions, const std::vector<std::string>& variables, const std::string& prefix="morefit_buffer_", float buffering_cost_threshold = 2.0) = 0;
    //evaluate compute graph for specific event/parameter combination, should nan if variable is not set
    virtual evalT eval(const std::vector<std::string>& variables, const std::vector<double>& values) const = 0;
    //draw graph in tikz using graphlib
    void draw(std::string filename) const
    {
      std::ofstream texfile;
      texfile.open(filename);
      texfile << "\\documentclass[tikz,border=10pt]{standalone}\n";
      texfile << "\\usetikzlibrary{graphdrawing}\n";
      texfile << "\\usetikzlibrary{graphs}\n";
      texfile << "\\usegdlibrary{trees}\n";
      texfile << "\\begin{document}\n";
      texfile << "\\begin{tikzpicture}[>=stealth, every node/.style={rectangle, rounded corners, fill=blue!25, draw, minimum size=0.75cm}]\n";
      texfile << "\\graph [tree layout, grow=down, fresh nodes, level distance=0.5in, sibling distance=0.5in]\n";
      texfile << "{\n";
      texfile << draw_node();
      texfile << "};\n";
      texfile << "\\end{tikzpicture}\n";
      texfile << "\\end{document}\n";
      texfile.close();
      return;
    }
    //draw node in tikz using graphlib    
    virtual std::string draw_node() const = 0;
    //check equality of graphs
    virtual bool operator==(const std::unique_ptr<ComputeGraphNode<kernelT, evalT>>& rhs) const = 0;
    //find first equal expression in list, return index
    virtual int find_in_list(std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>& list) const
    {
      int found_idx = -1;
      for (unsigned int i=0; i<list.size(); i++)
	if (this->operator==(list.at(i)))
	  {
	    found_idx = i;
	    break;
	  }
      return found_idx;
    }
  };

  //a numerical constant
  template<typename kernelT, typename evalT> 
  class ConstantNode: public ComputeGraphNode<kernelT, evalT> {
  public:
    evalT number_;      
    ConstantNode(evalT number)
      : number_(number)
    {
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> copy() const override 
    {
      return std::make_unique<ConstantNode<kernelT, evalT>>(number_);
    }
    virtual std::string get_kernel() const override
    {
      const bool hexfloat = false;
      std::stringstream str;
      if (hexfloat)
	str << std::hexfloat << number_;
      else
	str << std::scientific << std::setprecision(15) << number_;
      return str.str();
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> diff(std::string variable) const override
    {
      return std::make_unique<ConstantNode<kernelT, evalT>>(0);
    }        
    virtual void print(unsigned int level) const override
    {
      std::cout << std::string(level,'\t') << "ConstantNode: " << number_ << std::endl;
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> substitute(const std::vector<std::string>& variables, const std::vector<double>& values) const override
    {
      return std::make_unique<ConstantNode<kernelT, evalT>>(number_);
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> simplify() const override
    {
      return std::make_unique<ConstantNode<kernelT, evalT>>(number_);
    }
    virtual double cost() const override
    {
      return 0.1;
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> optimize_buffering_constant_terms(std::vector<std::string>& buffernames, std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>& bufferexpressions, const std::vector<std::string>& variables, const std::string& prefix="morefit_buffer_", float buffering_cost_threshold = 2.0) override
    {
      return std::make_unique<ConstantNode<kernelT, evalT>>(number_);
    }
    virtual evalT eval(const std::vector<std::string>& variables, const std::vector<double> & values) const override
    {
      return number_;
    }
    virtual std::string draw_node() const override
    {
      return "\""+std::to_string(number_)+"\"";//needed to escape negative numbers
    }
    //check equality of graphs
    virtual bool operator==(const std::unique_ptr<ComputeGraphNode<kernelT, evalT>>& rhs) const override
    {
      ConstantNode<kernelT, evalT>* rhs_ = dynamic_cast<ConstantNode<kernelT, evalT>*>(rhs.get());
      if (rhs_ != nullptr)
	return this->number_ == rhs_->number_;
      else
	return false;
    }
  };

  template<typename kernelT, typename evalT> 
  class SumNode: public ComputeGraphNode<kernelT, evalT> {
  public:
    SumNode(std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> children)
      : ComputeGraphNode<kernelT, evalT>(std::move(children))
    {
    }
    SumNode(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> childa, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> childb)
    {
      this->children_.emplace_back(std::move(childa));
      this->children_.emplace_back(std::move(childb));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> copy() const override
    {
      return std::make_unique<SumNode<kernelT, evalT>>(std::move(this->copy_children()));
    }
    virtual std::string get_kernel() const override
    {
      std::string result = "(";
      for (unsigned int i=0; i<this->children_.size(); i++)
	result += this->children_.at(i)->get_kernel() + (i == this->children_.size()-1 ? ")" : "+");
      return result;
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> diff(std::string variable) const override
    {
      if (!this->variable_in_tree(variable))
	return std::make_unique<ConstantNode<kernelT, evalT>>(0);
      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> reschildren;
      for (unsigned int i=0; i<this->children_.size(); i++)
	reschildren.emplace_back(std::move(this->children_.at(i)->diff(variable)));
      return std::make_unique<SumNode<kernelT, evalT>>(std::move(reschildren)); 
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> substitute(const std::vector<std::string>& variables, const std::vector<double>& values) const override
    {      
      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> reschildren;
      for (unsigned int i=0; i<this->children_.size(); i++)
	reschildren.emplace_back(std::move(this->children_.at(i)->substitute(variables, values)));      
      return std::make_unique<SumNode<kernelT, evalT>>(std::move(reschildren));      
    }
    //copies and tries to simplify tree
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> simplify() const override
    {
      //simplify children first
      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> reschildren;
      for (const std::unique_ptr<ComputeGraphNode<kernelT, evalT>> & child : this->children_)
	reschildren.emplace_back(std::move(child->simplify()));
      //combine sums of sums
      bool finished = false;
      while (!finished)//repeatedly insert summands from child sums
	{
	  int nsums = 0;
	  for (unsigned int i=0; i<reschildren.size(); i++)
	    if (dynamic_cast<SumNode<kernelT, evalT>*>(reschildren.at(i).get()) != nullptr)
	      {
		nsums++;
		//add sum children to reschildren
		for (const std::unique_ptr<ComputeGraphNode<kernelT, evalT>> & child : dynamic_cast<SumNode<kernelT, evalT>*>(reschildren.at(i).get())->children_)
		  reschildren.emplace_back(std::move(child->copy()));
		//remove sum
		reschildren.erase (reschildren.begin()+i);
		break;//and try again
	      }
	  if (nsums == 0)
	    finished = true;
	}
      //now simplify sum, check if multiple constants present
      evalT constant = 0.0;
      std::vector<unsigned int> indices;
      for (unsigned int i=0; i<reschildren.size(); i++)
	if (dynamic_cast<ConstantNode<kernelT, evalT>*>(reschildren.at(i).get()) != nullptr)
	  {
	    constant += dynamic_cast<ConstantNode<kernelT, evalT>*>(reschildren.at(i).get())->number_;
	    indices.push_back(i);
	  }
      std::reverse(indices.begin(), indices.end());
      if (indices.size() > 0)
	{
	  for (unsigned int i=0; i<indices.size(); i++)
	    {
	      unsigned int idx = indices.at(i);
	      reschildren.erase(reschildren.begin()+idx);
	    }
	  //insert common factor at the front
	  if (constant != 0.0 || reschildren.size() == 0)
	    reschildren.insert(reschildren.begin(), std::make_unique<ConstantNode<kernelT, evalT>>(constant));
	}
      if (reschildren.size() == 1)
	return reschildren.at(0)->copy();
      else
	return std::make_unique<SumNode<kernelT, evalT>>(std::move(reschildren));
    }    
    virtual void print(unsigned int level) const override
    {
      std::cout << std::string(level,'\t') << "SumNode with " << this->children_.size() << " children" << std::endl;
      for (unsigned int i=0; i<this->children_.size(); i++)
	this->children_.at(i)->print(level+1);
    }
    virtual double cost() const override
    {
      double c = 0.0;
      for (unsigned int i=0; i<this->children_.size(); i++)
	c += this->children_.at(i)->cost();
      return this->children_.size() + c;
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> optimize_buffering_constant_terms(std::vector<std::string>& buffernames, std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>& bufferexpressions, const std::vector<std::string>& variables, const std::string& prefix="morefit_buffer_", float buffering_cost_threshold = 2.0) override
    {
      if (cost() > buffering_cost_threshold)
	{
	  if (!this->contains_other_variables(variables))//expression does not depend on any variables besides the ones given, can optimise!
	    {
	      //first check if expression already exists in buffer list
	      int found_idx = this->find_in_list(bufferexpressions);
	      if (found_idx >= 0)
		return std::make_unique<VariableNode<kernelT, evalT>>(prefix+std::to_string(found_idx));
	      else
		{
		  //previous version
		  int buffer_idx = buffernames.size();
		  std::string buffername = prefix+std::to_string(buffer_idx);
		  buffernames.push_back(buffername);
		  bufferexpressions.emplace_back(this->copy());//just copy everything, no need for further optimisation
		  return std::make_unique<VariableNode<kernelT, evalT>>(buffername);
		}
	    }
	  else
	    {
	      //check all children, then optimize combination
	      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> reschildren;
	      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> optimizable_children;
	      double totalcost = 0.0;
	      for (const std::unique_ptr<ComputeGraphNode<kernelT, evalT>> & child : this->children_)
		{
		  if (!child->contains_other_variables(variables))
		    {
		      totalcost += child->cost();
		      optimizable_children.emplace_back(child->copy());		      
		    }
		  else
		    reschildren.emplace_back(std::move(child->optimize_buffering_constant_terms(buffernames, bufferexpressions, variables, prefix, buffering_cost_threshold)));
		}
	      if (optimizable_children.size() > 0 && totalcost > buffering_cost_threshold)
		{
		  if (optimizable_children.size() == 1)
		    {
		      int found_idx = optimizable_children.at(0)->find_in_list(bufferexpressions);
		      if (found_idx >= 0)
			reschildren.emplace_back(std::make_unique<VariableNode<kernelT, evalT>>(prefix+std::to_string(found_idx)));
		      else
			{
			  int buffer_idx = buffernames.size();
			  std::string buffername = prefix+std::to_string(buffer_idx);
			  buffernames.push_back(buffername);
			  bufferexpressions.emplace_back(std::move(optimizable_children.at(0)));
			  reschildren.emplace_back(std::make_unique<VariableNode<kernelT, evalT>>(buffername));
			}
		    }
		  else
		    {
		      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> opt_sum = std::make_unique<SumNode<kernelT,evalT>>(std::move(optimizable_children));
		      int found_idx = opt_sum->find_in_list(bufferexpressions);
		      if (found_idx >= 0)
			reschildren.emplace_back(std::make_unique<VariableNode<kernelT, evalT>>(prefix+std::to_string(found_idx)));
		      else
			{
			  int buffer_idx = buffernames.size();
			  std::string buffername = prefix+std::to_string(buffer_idx);
			  buffernames.push_back(buffername);
			  bufferexpressions.emplace_back(std::move(opt_sum));
			  reschildren.emplace_back(std::make_unique<VariableNode<kernelT, evalT>>(buffername));
			}		      
		    }
		}
	      else
		{
		  //add all optimizable children below cost threshold as is to reschildren
		  for (const std::unique_ptr<ComputeGraphNode<kernelT, evalT>> & child : optimizable_children)
		    reschildren.emplace_back(child->copy());
		}
	      return std::make_unique<SumNode<kernelT, evalT>>(std::move(reschildren));
	    }
	}
      else
	return this->copy();//copies also all children, but fine since we do not want to buffer this
    }
    virtual evalT eval(const std::vector<std::string>& variables, const std::vector<double> & values) const override
    {
      evalT result = 0.0;
      for (unsigned int i=0; i<this->children_.size(); i++)
	result += this->children_.at(i)->eval(variables, values);
      return result;
    }
    virtual std::string draw_node() const override
    {
      std::string result;
      result +=  "+ -> { ";
      for (unsigned int i=0; i<this->children_.size(); i++)
	result += this->children_.at(i)->draw_node() + (i == this->children_.size()-1 ? " }": ", ");
      return result;
    }
    //check equality of graphs
    virtual bool operator==(const std::unique_ptr<ComputeGraphNode<kernelT, evalT>>& rhs) const override
    {
      SumNode<kernelT, evalT>* rhs_ = dynamic_cast<SumNode<kernelT, evalT>*>(rhs.get());
      if (rhs_ != nullptr)
	{
	  if (this->children_.size() != rhs_->children_.size())
	    return false;
	  //have the rhs children been already matched?
	  std::vector<bool> rhs_matched(this->children_.size(), false);
	  for (unsigned int i=0; i<this->children_.size(); i++)
	    {
	      bool found_child = false;
	      for (unsigned int j=0; j<rhs_->children_.size(); j++)
		{		  
		  if (this->children_.at(i)->operator==(rhs_->children_.at(j)))
		    {
		      if (!(rhs_matched.at(j)))//check child has not already been matched
			{
			  rhs_matched.at(j) = true;
			  found_child = true;
			  break;
			}
		    }
		}
	      if (!found_child)		
		return false;
	    }
	  return true;
	}
      else
	return false;
    }
  };

  template<typename kernelT, typename evalT> 
  class ProdNode: public ComputeGraphNode<kernelT, evalT> {
  public:
    ProdNode(std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> children):
      ComputeGraphNode<kernelT, evalT>(std::move(children))
    {
    }
    ProdNode(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> childa, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> childb)
    {
      this->children_.emplace_back(std::move(childa));
      this->children_.emplace_back(std::move(childb));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> copy() const override
    {
      return std::make_unique<ProdNode<kernelT, evalT>>(std::move(this->copy_children()));
    }
    virtual std::string get_kernel() const override
    {
      std::string result = "(";
      for (unsigned int i=0; i<this->children_.size(); i++)
	result += this->children_.at(i)->get_kernel() + (i == this->children_.size()-1 ? ")" : "*");
      return result;
    }        
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> diff(std::string variable) const override
    {
      if (!this->variable_in_tree(variable))
	return std::make_unique<ConstantNode<kernelT, evalT>>(0);
      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> products;
      for (unsigned int i=0; i<this->children_.size(); i++)
	{
	  std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> prod_children;
	  for (unsigned int j=0; j<this->children_.size(); j++)
	    if (i!=j)
	      prod_children.push_back(this->children_.at(j)->copy());
	    else
	      prod_children.push_back(this->children_.at(j)->diff(variable));
	  products.emplace_back(std::make_unique<ProdNode<kernelT, evalT>>(std::move(prod_children)));
	}
      return std::make_unique<SumNode<kernelT, evalT>>(std::move(products));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> substitute(const std::vector<std::string>& variables, const std::vector<double>& values) const override
    {      
      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> reschildren;
      for (unsigned int i=0; i<this->children_.size(); i++)
	reschildren.emplace_back(std::move(this->children_.at(i)->substitute(variables, values)));      
      return std::make_unique<ProdNode<kernelT, evalT>>(std::move(reschildren));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> simplify() const override
    {
      //simplify children first
      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> reschildren;
      for (const std::unique_ptr<ComputeGraphNode<kernelT, evalT>> & child : this->children_)
	reschildren.emplace_back(std::move(child->simplify()));
      //combine products of products
      bool finished = false;
      while (!finished)//repeatedly insert factors from child products
	{
	  int nproducts = 0;
	  for (unsigned int i=0; i<reschildren.size(); i++)
	    if (dynamic_cast<ProdNode<kernelT, evalT>*>(reschildren.at(i).get()) != nullptr)
	      {
		nproducts++;
		//add product children to reschildren
		for (const std::unique_ptr<ComputeGraphNode<kernelT, evalT>> & child : dynamic_cast<ProdNode<kernelT, evalT>*>(reschildren.at(i).get())->children_)
		  reschildren.emplace_back(std::move(child->copy()));
		//remove product
		reschildren.erase (reschildren.begin()+i);
		break;//and try again
	      }
	  if (nproducts == 0)
	    finished = true;
	}
      //now simplify/combined product of constants
      evalT constant = 1.0;
      std::vector<unsigned int> indices;
      for (unsigned int i=0; i<reschildren.size(); i++)
	if (dynamic_cast<ConstantNode<kernelT, evalT>*>(reschildren.at(i).get()) != nullptr)
	  {
	    constant *= dynamic_cast<ConstantNode<kernelT, evalT>*>(reschildren.at(i).get())->number_;
	    indices.push_back(i);
	  }
      if (constant == 0.0)
	return std::make_unique<ConstantNode<kernelT, evalT>>(constant);
      std::reverse(indices.begin(), indices.end());
      if (indices.size() > 0)
	{
	  for (unsigned int i=0; i<indices.size(); i++)
	    {
	      unsigned int idx = indices.at(i);
	      reschildren.erase(reschildren.begin()+idx);
	    }
	  //insert common factor at the front
	  if (constant != 1.0 || reschildren.size() == 0)
	    reschildren.insert(reschildren.begin(), std::make_unique<ConstantNode<kernelT, evalT>>(constant));
	}
      if (reschildren.size() == 1)
	return reschildren.at(0)->copy();
      else
	return std::make_unique<ProdNode<kernelT, evalT>>(std::move(reschildren));
    }
    virtual void print(unsigned int level) const override
    {
      std::cout << std::string(level,'\t') << "ProdNode with " << this->children_.size() << " children" << std::endl;
      for (unsigned int i=0; i<this->children_.size(); i++)
	this->children_.at(i)->print(level+1);
    }
    virtual double cost() const override
    {
      double c = 0.0;
      for (unsigned int i=0; i<this->children_.size(); i++)
	c += this->children_.at(i)->cost();
      return 2.0*this->children_.size() + c;
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> optimize_buffering_constant_terms(std::vector<std::string>& buffernames, std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>& bufferexpressions, const std::vector<std::string>& variables, const std::string& prefix="morefit_buffer_", float buffering_cost_threshold = 2.0) override
    {
      if (cost() > buffering_cost_threshold)
	{
	  if (!this->contains_other_variables(variables))//expression does not depend on any variables besides the ones given, can optimise!
	    {
	      int found_idx = this->find_in_list(bufferexpressions);
	      if (found_idx >= 0)
		return std::make_unique<VariableNode<kernelT, evalT>>(prefix+std::to_string(found_idx));
	      else
		{
		  int buffer_idx = buffernames.size();
		  std::string buffername = prefix+std::to_string(buffer_idx);
		  buffernames.push_back(buffername);
		  bufferexpressions.emplace_back(this->copy());//just copy everything, no need for further optimisation
		  return std::make_unique<VariableNode<kernelT, evalT>>(buffername);
		}
	    }
	  else
	    {
	      //check all children, then optimize combination
	      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> reschildren;
	      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> optimizable_children;
	      double totalcost = 0.0;
	      for (const std::unique_ptr<ComputeGraphNode<kernelT, evalT>> & child : this->children_)
		{
		  if (!child->contains_other_variables(variables))
		    {
		      totalcost += child->cost();
		      optimizable_children.emplace_back(child->copy());
		    }
		  else
		    reschildren.emplace_back(std::move(child->optimize_buffering_constant_terms(buffernames, bufferexpressions, variables, prefix, buffering_cost_threshold)));
		}
	      if (optimizable_children.size() > 0 && totalcost > buffering_cost_threshold)
		{
		  if (optimizable_children.size() == 1)
		    {
		      int found_idx = optimizable_children.at(0)->find_in_list(bufferexpressions);
		      if (found_idx >= 0)
			reschildren.emplace_back(std::make_unique<VariableNode<kernelT, evalT>>(prefix+std::to_string(found_idx)));
		      else
			{
			  int buffer_idx = buffernames.size();
			  std::string buffername = prefix+std::to_string(buffer_idx);
			  buffernames.push_back(buffername);
			  bufferexpressions.emplace_back(std::move(optimizable_children.at(0)));
			  reschildren.emplace_back(std::make_unique<VariableNode<kernelT, evalT>>(buffername));
			}
		    }
		  else
		    {
		      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> opt_prod = std::make_unique<ProdNode<kernelT,evalT>>(std::move(optimizable_children));
		      int found_idx = opt_prod->find_in_list(bufferexpressions);
		      if (found_idx >= 0)
			reschildren.emplace_back(std::make_unique<VariableNode<kernelT, evalT>>(prefix+std::to_string(found_idx)));
		      else
			{
			  int buffer_idx = buffernames.size();
			  std::string buffername = prefix+std::to_string(buffer_idx);
			  buffernames.push_back(buffername);
			  bufferexpressions.emplace_back(std::move(opt_prod));
			  reschildren.emplace_back(std::make_unique<VariableNode<kernelT, evalT>>(buffername));
			}		      
		    }
		}
	      else
		{
		  //add all optimizable children below cost threshold as is to reschildren
		  for (const std::unique_ptr<ComputeGraphNode<kernelT, evalT>> & child : optimizable_children)
		    reschildren.emplace_back(child->copy());
		}
	      return std::make_unique<ProdNode<kernelT, evalT>>(std::move(reschildren));
	    }
	}
      else
	return this->copy();//copies also all children, but fine since we do not want to buffer this
    }
    virtual evalT eval(const std::vector<std::string>& variables, const std::vector<double> & values) const override
    {
      evalT result = 1.0;
      for (unsigned int i=0; i<this->children_.size(); i++)
	result *= this->children_.at(i)->eval(variables, values);
      return result;
    }
    virtual std::string draw_node() const override
    {
      std::string result;
      result +=  "* -> { ";
      for (unsigned int i=0; i<this->children_.size(); i++)
	result += this->children_.at(i)->draw_node() + (i == this->children_.size()-1 ? " }": ", ");
      return result;
    }
    //check equality of graphs
    virtual bool operator==(const std::unique_ptr<ComputeGraphNode<kernelT, evalT>>& rhs) const override
    {
      ProdNode<kernelT, evalT>* rhs_ = dynamic_cast<ProdNode<kernelT, evalT>*>(rhs.get());
      if (rhs_ != nullptr)
	{
	  if (this->children_.size() != rhs_->children_.size())
	    return false;
	  //have the rhs children been already matched?
	  std::vector<bool> rhs_matched(this->children_.size(), false);
	  for (unsigned int i=0; i<this->children_.size(); i++)
	    {
	      bool found_child = false;
	      for (unsigned int j=0; j<rhs_->children_.size(); j++)
		{		  
		  if (this->children_.at(i)->operator==(rhs_->children_.at(j)))
		    {
		      if (!(rhs_matched.at(j)))//check child has not already been matched
			{
			  rhs_matched.at(j) = true;
			  found_child = true;
			  break;
			}
		    }
		}
	      if (!found_child)
		return false;
	    }
	  return true;
	}
      else
	return false;
    }
  };

  template<typename kernelT, typename evalT> 
  class ExpNode: public ComputeGraphNode<kernelT, evalT> {
  public:
    ExpNode(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> child):
      ComputeGraphNode<kernelT, evalT>(std::move(child))
    {
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> copy() const override
    {
      return std::make_unique<ExpNode<kernelT, evalT>>(std::move(this->copy_children().at(0)));
    }
    virtual std::string get_kernel() const override
    {      
      return "exp(" + this->children_.at(0)->get_kernel() + ")";
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> diff(std::string variable) const override
    {      
      if (!this->variable_in_tree(variable))
	return std::make_unique<ConstantNode<kernelT, evalT>>(0);
      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> reschildren;
      reschildren.emplace_back(std::move(this->children_.at(0)->diff(variable)));
      reschildren.emplace_back(std::move(this->copy()));
      return std::make_unique<ProdNode<kernelT, evalT>>(std::move(reschildren));
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> substitute(const std::vector<std::string>& variables, const std::vector<double>& values) const override
    {
      return std::make_unique<ExpNode<kernelT, evalT>>(std::move(this->children_.at(0)->substitute(variables, values)));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> simplify() const override
    {
      std::unique_ptr child = this->children_.at(0)->simplify();
      if (dynamic_cast<ConstantNode<kernelT, evalT>*>(child.get()) != nullptr)
	{
	  evalT constant = exp(dynamic_cast<ConstantNode<kernelT, evalT>*>(child.get())->number_);
	  return std::make_unique<ConstantNode<kernelT, evalT>>(constant);
	}
      else if (dynamic_cast<LogNode<kernelT, evalT>*>(child.get()) != nullptr)
	{
	  return dynamic_cast<LogNode<kernelT, evalT>*>(child.get())->children_.at(0)->copy();
	}
      else
	return std::make_unique<ExpNode<kernelT, evalT>>(std::move(child));
    }
    virtual void print(unsigned int level) const override
    {
      std::cout << std::string(level,'\t') << "ExpNode with " << this->children_.size() << " children" << std::endl;
      for (unsigned int i=0; i<this->children_.size(); i++)
	this->children_.at(i)->print(level+1);
    }
    virtual double cost() const override
    {
      double c = this->children_.at(0)->cost();
      return 4.0 + c;
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> optimize_buffering_constant_terms(std::vector<std::string>& buffernames, std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>& bufferexpressions, const std::vector<std::string>& variables, const std::string& prefix="morefit_buffer_", float buffering_cost_threshold = 2.0) override
    {
      if (cost() > buffering_cost_threshold)
	{
	  if (!this->contains_other_variables(variables))//expression does not depend on any variables besides the ones given, can optimise!
	    {
	      int found_idx = this->find_in_list(bufferexpressions);
	      if (found_idx >= 0)
		return std::make_unique<VariableNode<kernelT, evalT>>(prefix+std::to_string(found_idx));
	      else
		{	      
		  int buffer_idx = buffernames.size();
		  std::string buffername = prefix+std::to_string(buffer_idx);
		  buffernames.push_back(buffername);
		  bufferexpressions.emplace_back(this->copy());//just copy everything, no need for further optimisation
		  return std::make_unique<VariableNode<kernelT, evalT>>(buffername);
		}
	    }
	  else
	    return std::make_unique<ExpNode<kernelT, evalT>>(std::move(this->children_.at(0)->optimize_buffering_constant_terms(buffernames, bufferexpressions, variables, prefix, buffering_cost_threshold)));
	}
      else
	return this->copy();//copies also all children, but fine since we do not want to buffer this
    }
    virtual evalT eval(const std::vector<std::string>& variables, const std::vector<double> & values) const override
    {
      return exp(this->children_.at(0)->eval(variables, values));
    }
    virtual std::string draw_node() const override
    {
      std::string result;
      result +=  "exp -> { ";
      result += this->children_.at(0)->draw_node() + "}";
      return result;
    }
    //check equality of graphs
    virtual bool operator==(const std::unique_ptr<ComputeGraphNode<kernelT, evalT>>& rhs) const override
    {
      ExpNode<kernelT, evalT>* rhs_ = dynamic_cast<ExpNode<kernelT, evalT>*>(rhs.get());
      if (rhs_ != nullptr)
	return this->children_.at(0)->operator==(rhs_->children_.at(0));
      else
	return false;
    }
  };

  template<typename kernelT, typename evalT> 
  class InvNode: public ComputeGraphNode<kernelT, evalT> {
  public:
    InvNode(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> child):
      ComputeGraphNode<kernelT, evalT>(std::move(child))
    {
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> copy() const override
    {
      return std::make_unique<InvNode<kernelT, evalT>>(std::move(this->copy_children().at(0)));
    }
    virtual std::string get_kernel() const override
    {      
      return "1.0/(" + this->children_.at(0)->get_kernel() + ")";
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> diff(std::string variable) const override
    {
      if (!this->variable_in_tree(variable))
	return std::make_unique<ConstantNode<kernelT, evalT>>(0);
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> inner = this->children_.at(0)->diff(variable);
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> outer = this->children_.at(0)->copy();
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> outer2 = this->children_.at(0)->copy();
      std::unique_ptr<InvNode<kernelT, evalT>> invsq = std::make_unique<InvNode<kernelT, evalT>>(std::make_unique<ProdNode<kernelT, evalT>>(make_vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>(std::move(outer),std::move(outer2))));      
      return std::make_unique<NegNode<kernelT, evalT>>(std::make_unique<ProdNode<kernelT, evalT>>(make_vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>(std::move(inner), std::move(invsq))));
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> substitute(const std::vector<std::string>& variables, const std::vector<double>& values) const override
    {
      return std::make_unique<InvNode<kernelT, evalT>>(std::move(this->children_.at(0)->substitute(variables, values)));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> simplify() const override
    {
      std::unique_ptr child = this->children_.at(0)->simplify();
      if (dynamic_cast<ConstantNode<kernelT, evalT>*>(child.get()) != nullptr)
	{
	  evalT constant = 1.0/(dynamic_cast<ConstantNode<kernelT, evalT>*>(child.get())->number_);
	  return std::make_unique<ConstantNode<kernelT, evalT>>(constant);
	}
      else if (dynamic_cast<InvNode<kernelT, evalT>*>(child.get()) != nullptr)
	{
	  return dynamic_cast<InvNode<kernelT, evalT>*>(child.get())->children_.at(0)->copy();
	}
      else
	return std::make_unique<InvNode<kernelT, evalT>>(std::move(child));
    }
    virtual void print(unsigned int level) const override
    {
      std::cout << std::string(level,'\t') << "InvNode with " << this->children_.size() << " children" << std::endl;
      for (unsigned int i=0; i<this->children_.size(); i++)
	this->children_.at(i)->print(level+1);
    }
    virtual double cost() const override
    {
      double c = this->children_.at(0)->cost();
      return 4.0 + c;
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> optimize_buffering_constant_terms(std::vector<std::string>& buffernames, std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>& bufferexpressions, const std::vector<std::string>& variables, const std::string& prefix="morefit_buffer_", float buffering_cost_threshold = 2.0) override
    {
      if (cost() > buffering_cost_threshold)
	{
	  if (!this->contains_other_variables(variables))//expression does not depend on any variables besides the ones given, can optimise!
	    {
	      int found_idx = this->find_in_list(bufferexpressions);
	      if (found_idx >= 0)
		return std::make_unique<VariableNode<kernelT, evalT>>(prefix+std::to_string(found_idx));
	      else
		{	      
		  int buffer_idx = buffernames.size();
		  std::string buffername = prefix+std::to_string(buffer_idx);
		  buffernames.push_back(buffername);
		  bufferexpressions.emplace_back(this->copy());//just copy everything, no need for further optimisation
		  return std::make_unique<VariableNode<kernelT, evalT>>(buffername);
		}
	    }
	  else
	    return std::make_unique<InvNode<kernelT, evalT>>(std::move(this->children_.at(0)->optimize_buffering_constant_terms(buffernames, bufferexpressions, variables, prefix, buffering_cost_threshold)));
	}
      else
	return this->copy();//copies also all children, but fine since we do not want to buffer this
    }
    virtual evalT eval(const std::vector<std::string>& variables, const std::vector<double> & values) const override
    {
      return 1.0/(this->children_.at(0)->eval(variables, values));
    }
    virtual std::string draw_node() const override
    {
      std::string result;
      result +=  "inv -> { ";
      result += this->children_.at(0)->draw_node() + "}";
      return result;
    }
    //check equality of graphs
    virtual bool operator==(const std::unique_ptr<ComputeGraphNode<kernelT, evalT>>& rhs) const override
    {
      InvNode<kernelT, evalT>* rhs_ = dynamic_cast<InvNode<kernelT, evalT>*>(rhs.get());
      if (rhs_ != nullptr)
	return this->children_.at(0)->operator==(rhs_->children_.at(0));
      else
	return false;
    }
  };

  template<typename kernelT, typename evalT> 
  class LogNode: public ComputeGraphNode<kernelT, evalT> {
  public:
    LogNode(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> child):
      ComputeGraphNode<kernelT, evalT>(std::move(child))
    {
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> copy() const override
    {
      return std::make_unique<LogNode<kernelT, evalT>>(std::move(this->copy_children().at(0)));
    }
    virtual std::string get_kernel() const override
    {      
      return "log(" + this->children_.at(0)->get_kernel() + ")";
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> diff(std::string variable) const override
    {
      if (!this->variable_in_tree(variable))
	return std::make_unique<ConstantNode<kernelT, evalT>>(0);
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> inner = this->children_.at(0)->diff(variable);
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> arg = this->children_.at(0)->copy();
      std::unique_ptr<InvNode<kernelT, evalT>> outer = std::make_unique<InvNode<kernelT, evalT>>(std::move(arg));
      return std::make_unique<ProdNode<kernelT, evalT>>(make_vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>(std::move(inner), std::move(outer)));
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> substitute(const std::vector<std::string>& variables, const std::vector<double>& values) const override
    {
      return std::make_unique<LogNode<kernelT, evalT>>(std::move(this->children_.at(0)->substitute(variables, values)));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> simplify() const override
    {
      std::unique_ptr child = this->children_.at(0)->simplify();
      if (dynamic_cast<ConstantNode<kernelT, evalT>*>(child.get()) != nullptr)
	{
	  evalT constant = log(dynamic_cast<ConstantNode<kernelT, evalT>*>(child.get())->number_);
	  return std::make_unique<ConstantNode<kernelT, evalT>>(constant);
	}
      else if (dynamic_cast<ExpNode<kernelT, evalT>*>(child.get()) != nullptr)
	{
	  return dynamic_cast<ExpNode<kernelT, evalT>*>(child.get())->children_.at(0)->copy();
	}
      else
	return std::make_unique<LogNode<kernelT, evalT>>(std::move(child));
    }
    virtual void print(unsigned int level) const override
    {
      std::cout << std::string(level,'\t') << "LogNode with " << this->children_.size() << " children" << std::endl;
      for (unsigned int i=0; i<this->children_.size(); i++)
	this->children_.at(i)->print(level+1);
    }
    virtual double cost() const override
    {
      double c = this->children_.at(0)->cost();
      return 4.0 + c;
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> optimize_buffering_constant_terms(std::vector<std::string>& buffernames, std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>& bufferexpressions, const std::vector<std::string>& variables, const std::string& prefix="morefit_buffer_", float buffering_cost_threshold = 2.0) override
    {
      if (cost() > buffering_cost_threshold)
	{
	  if (!this->contains_other_variables(variables))//expression does not depend on any variables besides the ones given, can optimise!
	    {
	      int found_idx = this->find_in_list(bufferexpressions);
	      if (found_idx >= 0)
		return std::make_unique<VariableNode<kernelT, evalT>>(prefix+std::to_string(found_idx));
	      else
		{	      
		  int buffer_idx = buffernames.size();
		  std::string buffername = prefix+std::to_string(buffer_idx);
		  buffernames.push_back(buffername);
		  bufferexpressions.emplace_back(this->copy());//just copy everything, no need for further optimisation
		  return std::make_unique<VariableNode<kernelT, evalT>>(buffername);
		}
	    }
	  else
	    return std::make_unique<LogNode<kernelT, evalT>>(std::move(this->children_.at(0)->optimize_buffering_constant_terms(buffernames, bufferexpressions, variables, prefix, buffering_cost_threshold)));
	}
      else
	return this->copy();//copies also all children, but fine since we do not want to buffer this
    }
    virtual evalT eval(const std::vector<std::string>& variables, const std::vector<double> & values) const override
    {
      return log(this->children_.at(0)->eval(variables, values));
    }
    virtual std::string draw_node() const override
    {
      std::string result;
      result +=  "log -> { ";
      result += this->children_.at(0)->draw_node() + "}";
      return result;
    }
    //check equality of graphs
    virtual bool operator==(const std::unique_ptr<ComputeGraphNode<kernelT, evalT>>& rhs) const override
    {
      LogNode<kernelT, evalT>* rhs_ = dynamic_cast<LogNode<kernelT, evalT>*>(rhs.get());
      if (rhs_ != nullptr)
	return this->children_.at(0)->operator==(rhs_->children_.at(0));
      else
	return false;
    }
  };



  template<typename kernelT, typename evalT> 
  class PowNode: public ComputeGraphNode<kernelT, evalT> {
  public:
    PowNode(std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> children)
      : ComputeGraphNode<kernelT, evalT>(std::move(children))
    {
    }
    PowNode(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> base, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> exponent)
    {
      this->children_.emplace_back(std::move(base));
      this->children_.emplace_back(std::move(exponent));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> copy() const override
    {
      return std::make_unique<PowNode<kernelT, evalT>>(std::move(this->copy_children()));
    }
    virtual std::string get_kernel() const override
    {
      std::string result = "pow(";
      result += this->children_.at(0)->get_kernel();
      result += ",";
      result += this->children_.at(1)->get_kernel();
      result += ")";
      return result;
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> diff(std::string variable) const override
    {
      if (!this->variable_in_tree(variable))
	return std::make_unique<ConstantNode<kernelT, evalT>>(0);
      if (!this->children_.at(1)->variable_in_tree(variable))//exponent does not depend on variable
	{//Y*pow(x,y-1)*dx/dvar	  
	  std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> children;
	  children.emplace_back(this->children_.at(1)->copy());

	  std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> exponent_minus_one_children;
	  exponent_minus_one_children.emplace_back(this->children_.at(1)->copy());
	  exponent_minus_one_children.emplace_back(std::make_unique<ConstantNode<kernelT, evalT>>(-1.0));
	  std::unique_ptr<SumNode<kernelT, evalT>> exponent_minus_one(std::make_unique<SumNode<kernelT, evalT>>(std::move(exponent_minus_one_children)));
	  
	  children.emplace_back(std::make_unique<PowNode<kernelT, evalT>>(this->children_.at(0)->copy(),std::move(exponent_minus_one)));
	  children.emplace_back(this->children_.at(0)->diff(variable));
	  return std::make_unique<ProdNode<kernelT, evalT>>(std::move(children));
	}
      else if (!this->children_.at(0)->variable_in_tree(variable))//base does not depend on variable
	{//pow(x,y)*log(x)*dy/dvar
	  std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> children;
	  children.emplace_back(this->copy());
	  children.emplace_back(std::make_unique<LogNode<kernelT, evalT>>(this->children_.at(0)->copy()));
	  children.emplace_back(this->children_.at(1)->diff(variable));
	  return std::make_unique<ProdNode<kernelT, evalT>>(std::move(children));	  
	}
      else
	{//Y*pow(x,y-1)*dx/dvar	 +  pow(x,y)*log(x)*dy/dvar
	  std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> children_first;
	  children_first.emplace_back(this->children_.at(1)->copy());

	  std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> exponent_minus_one_children;
	  exponent_minus_one_children.emplace_back(this->children_.at(1)->copy());
	  exponent_minus_one_children.emplace_back(std::make_unique<ConstantNode<kernelT, evalT>>(-1.0));
	  std::unique_ptr<SumNode<kernelT, evalT>> exponent_minus_one(std::make_unique<SumNode<kernelT, evalT>>(std::move(exponent_minus_one_children)));
	  
	  children_first.emplace_back(std::make_unique<PowNode<kernelT, evalT>>(this->children_.at(0)->copy(),std::move(exponent_minus_one)));
	  children_first.emplace_back(this->children_.at(0)->diff(variable));

	  std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> children_second;
	  children_second.emplace_back(this->copy());
	  children_second.emplace_back(std::make_unique<LogNode<kernelT, evalT>>(this->children_.at(0)->copy()));
	  children_second.emplace_back(this->children_.at(1)->diff(variable));

	  std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> children_sum;
	  children_sum.emplace_back(std::make_unique<ProdNode<kernelT, evalT>>(std::move(children_first)));
	  children_sum.emplace_back(std::make_unique<ProdNode<kernelT, evalT>>(std::move(children_second)));	  
	  return std::make_unique<SumNode<kernelT, evalT>>(std::move(children_sum));
	}
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> substitute(const std::vector<std::string>& variables, const std::vector<double>& values) const override
    {      
      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> reschildren;
      for (unsigned int i=0; i<this->children_.size(); i++)
	reschildren.emplace_back(std::move(this->children_.at(i)->substitute(variables, values)));      
      return std::make_unique<PowNode<kernelT, evalT>>(std::move(reschildren));      
    }
    //copies and tries to simplify tree
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> simplify() const override
    {
      //simplify children first
      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> reschildren;
      for (const std::unique_ptr<ComputeGraphNode<kernelT, evalT>> & child : this->children_)
	reschildren.emplace_back(std::move(child->simplify()));
      //if both children are constants, evaluate power
      if (dynamic_cast<ConstantNode<kernelT, evalT>*>(reschildren.at(0).get()) != nullptr && dynamic_cast<ConstantNode<kernelT, evalT>*>(reschildren.at(1).get()) != nullptr)
	{
	  evalT result = pow(dynamic_cast<ConstantNode<kernelT, evalT>*>(reschildren.at(0).get())->number_, dynamic_cast<ConstantNode<kernelT, evalT>*>(reschildren.at(1).get())->number_);
	  return std::make_unique<ConstantNode<kernelT, evalT>>(result);
	}
      else
	return std::make_unique<PowNode<kernelT, evalT>>(std::move(reschildren));
    }    
    virtual void print(unsigned int level) const override
    {
      std::cout << std::string(level,'\t') << "PowNode with " << this->children_.size() << " children" << std::endl;
      for (unsigned int i=0; i<this->children_.size(); i++)
	this->children_.at(i)->print(level+1);
    }
    virtual double cost() const override
    {
      double c = this->children_.at(0)->cost();//base
      c += this->children_.at(1)->cost();//exponent
      return 4.0 + c;
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> optimize_buffering_constant_terms(std::vector<std::string>& buffernames, std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>& bufferexpressions, const std::vector<std::string>& variables, const std::string& prefix="morefit_buffer_", float buffering_cost_threshold = 2.0) override
    {
      if (cost() > buffering_cost_threshold)
	{
	  if (!this->contains_other_variables(variables))//expression does not depend on any variables besides the ones given, can optimise!
	    {
	      int found_idx = this->find_in_list(bufferexpressions);
	      if (found_idx >= 0)
		return std::make_unique<VariableNode<kernelT, evalT>>(prefix+std::to_string(found_idx));
	      else
		{	      
		  int buffer_idx = buffernames.size();
		  std::string buffername = prefix+std::to_string(buffer_idx);
		  buffernames.push_back(buffername);
		  bufferexpressions.emplace_back(this->copy());//just copy everything, no need for further optimisation
		  return std::make_unique<VariableNode<kernelT, evalT>>(buffername);
		}
	    }
	  else
	    {
	      //loop over children, at the end return sum of children
	      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> reschildren;
	      for (const std::unique_ptr<ComputeGraphNode<kernelT, evalT>> & child : this->children_)
		reschildren.emplace_back(std::move(child->optimize_buffering_constant_terms(buffernames, bufferexpressions, variables, prefix, buffering_cost_threshold)));
	      return std::make_unique<PowNode<kernelT, evalT>>(std::move(reschildren));
	    }
	}
      else
	return this->copy();//copies also all children, but fine since we do not want to buffer this
    }
    virtual evalT eval(const std::vector<std::string>& variables, const std::vector<double> & values) const override
    {
      evalT result = pow(this->children_.at(0)->eval(variables, values), this->children_.at(1)->eval(variables, values));
      return result;
    }
    virtual std::string draw_node() const override
    {
      std::string result;
      result +=  "pow -> { ";
      result += this->children_.at(0)->draw_node() + ", " + this->children_.at(1)->draw_node() + "}";
      return result;
    }
    //check equality of graphs
    virtual bool operator==(const std::unique_ptr<ComputeGraphNode<kernelT, evalT>>& rhs) const override
    {
      PowNode<kernelT, evalT>* rhs_ = dynamic_cast<PowNode<kernelT, evalT>*>(rhs.get());
      if (rhs_ != nullptr)
	return (this->children_.at(0)->operator==(rhs_->children_.at(0)) && this->children_.at(1)->operator==(rhs_->children_.at(1)));
      else
	return false;
    }
  };


  template<typename kernelT, typename evalT> 
  class NegNode: public ComputeGraphNode<kernelT, evalT> {
  public:
    NegNode(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> child):
      ComputeGraphNode<kernelT, evalT>(std::move(child))
    {
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> copy() const override
    {
      return std::make_unique<NegNode<kernelT, evalT>>(std::move(this->copy_children().at(0)));
    }
    virtual std::string get_kernel() const override
    {      
      return "-(" + this->children_.at(0)->get_kernel() + ")";
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> diff(std::string variable) const override
    {
      if (!this->variable_in_tree(variable))
	return std::make_unique<ConstantNode<kernelT, evalT>>(0);
      return std::make_unique<NegNode<kernelT, evalT>>(this->children_.at(0)->diff(variable));
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> substitute(const std::vector<std::string>& variables, const std::vector<double>& values) const override
    {
      return std::make_unique<NegNode<kernelT, evalT>>(std::move(this->children_.at(0)->substitute(variables, values)));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> simplify() const override
    {
      std::unique_ptr child = this->children_.at(0)->simplify();
      if (dynamic_cast<ConstantNode<kernelT, evalT>*>(child.get()) != nullptr)
	{
	  evalT constant = -(dynamic_cast<ConstantNode<kernelT, evalT>*>(child.get())->number_);
	  return std::make_unique<ConstantNode<kernelT, evalT>>(constant);
	}
      else if (dynamic_cast<NegNode<kernelT, evalT>*>(child.get()) != nullptr)
	{
	  return dynamic_cast<NegNode<kernelT, evalT>*>(child.get())->children_.at(0)->copy();
	}
      else
	return std::make_unique<NegNode<kernelT, evalT>>(std::move(child));
    }
    virtual void print(unsigned int level) const override
    {
      std::cout << std::string(level,'\t') << "NegNode with " << this->children_.size() << " children" << std::endl;
      for (unsigned int i=0; i<this->children_.size(); i++)
	this->children_.at(i)->print(level+1);
    }
    virtual double cost() const override
    {
      double c = this->children_.at(0)->cost();
      return 1.0 + c;
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> optimize_buffering_constant_terms(std::vector<std::string>& buffernames, std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>& bufferexpressions, const std::vector<std::string>& variables, const std::string& prefix="morefit_buffer_", float buffering_cost_threshold = 2.0) override
    {
      if (cost() > buffering_cost_threshold)
	{
	  if (!this->contains_other_variables(variables))//expression does not depend on any variables besides the ones given, can optimise!
	    {
	      int found_idx = this->find_in_list(bufferexpressions);
	      if (found_idx >= 0)
		return std::make_unique<VariableNode<kernelT, evalT>>(prefix+std::to_string(found_idx));
	      else
		{
		  int buffer_idx = buffernames.size();
		  std::string buffername = prefix+std::to_string(buffer_idx);
		  buffernames.push_back(buffername);
		  bufferexpressions.emplace_back(this->copy());//just copy everything, no need for further optimisation
		  return std::make_unique<VariableNode<kernelT, evalT>>(buffername);
		}
	    }
	  else
	    return std::make_unique<NegNode<kernelT, evalT>>(std::move(this->children_.at(0)->optimize_buffering_constant_terms(buffernames, bufferexpressions, variables, prefix, buffering_cost_threshold)));
	}
      else
	return this->copy();//copies also all children, but fine since we do not want to buffer this
    }
    virtual evalT eval(const std::vector<std::string>& variables, const std::vector<double> & values) const override
    {
      return -(this->children_.at(0)->eval(variables, values));
    }
    virtual std::string draw_node() const override
    {
      std::string result;
      result +=  "\"-\" -> { ";
      result += this->children_.at(0)->draw_node() + "}";
      return result;
    }
    //check equality of graphs
    virtual bool operator==(const std::unique_ptr<ComputeGraphNode<kernelT, evalT>>& rhs) const override
    {
      NegNode<kernelT, evalT>* rhs_ = dynamic_cast<NegNode<kernelT, evalT>*>(rhs.get());
      if (rhs_ != nullptr)
	return this->children_.at(0)->operator==(rhs_->children_.at(0));
      else
	return false;
    }
  };

  template<typename kernelT, typename evalT> //could potentially move special treatment for exponent = 0.5 to PowNode
  class SqrtNode: public ComputeGraphNode<kernelT, evalT> {
  public:
    SqrtNode(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> child):
      ComputeGraphNode<kernelT, evalT>(std::move(child))
    {
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> copy() const override
    {
      return std::make_unique<SqrtNode<kernelT, evalT>>(std::move(this->copy_children().at(0)));
    }
    virtual std::string get_kernel() const override
    {      
      return "sqrt(" + this->children_.at(0)->get_kernel() + ")";
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> diff(std::string variable) const override
    {
      if (!this->variable_in_tree(variable))
	return std::make_unique<ConstantNode<kernelT, evalT>>(0);
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> inner = this->children_.at(0)->diff(variable);
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> arg = this->children_.at(0)->copy();
      std::unique_ptr<ProdNode<kernelT, evalT>> outer = std::make_unique<ProdNode<kernelT, evalT>>(std::make_unique<ConstantNode<kernelT, evalT>>(0.5), std::make_unique<InvNode<kernelT, evalT>>(std::make_unique<SqrtNode<kernelT, evalT>>(std::move(arg))) );
      return std::make_unique<ProdNode<kernelT, evalT>>(std::move(inner), std::move(outer));
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> substitute(const std::vector<std::string>& variables, const std::vector<double>& values) const override
    {
      return std::make_unique<SqrtNode<kernelT, evalT>>(std::move(this->children_.at(0)->substitute(variables, values)));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> simplify() const override
    {
      std::unique_ptr child = this->children_.at(0)->simplify();
      if (dynamic_cast<ConstantNode<kernelT, evalT>*>(child.get()) != nullptr)
	{
	  evalT constant = sqrt(dynamic_cast<ConstantNode<kernelT, evalT>*>(child.get())->number_);
	  return std::make_unique<ConstantNode<kernelT, evalT>>(constant);
	}
      else if (dynamic_cast<SqrtNode<kernelT, evalT>*>(child.get()) != nullptr)
	{
	  return std::make_unique<PowNode<kernelT, evalT>>(dynamic_cast<SqrtNode<kernelT, evalT>*>(child.get())->children_.at(0)->copy(), std::make_unique<ConstantNode<kernelT, evalT>>(0.25));
	}
      else
	return std::make_unique<SqrtNode<kernelT, evalT>>(std::move(child));
    }
    virtual void print(unsigned int level) const override
    {
      std::cout << std::string(level,'\t') << "SqrtNode with " << this->children_.size() << " children" << std::endl;
      for (unsigned int i=0; i<this->children_.size(); i++)
	this->children_.at(i)->print(level+1);
    }
    virtual double cost() const override
    {
      double c = this->children_.at(0)->cost();
      return 2.0 + c;
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> optimize_buffering_constant_terms(std::vector<std::string>& buffernames, std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>& bufferexpressions, const std::vector<std::string>& variables, const std::string& prefix="morefit_buffer_", float buffering_cost_threshold = 2.0) override
    {
      if (cost() > buffering_cost_threshold)
	{
	  if (!this->contains_other_variables(variables))//expression does not depend on any variables besides the ones given, can optimise!
	    {
	      int found_idx = this->find_in_list(bufferexpressions);
	      if (found_idx >= 0)
		return std::make_unique<VariableNode<kernelT, evalT>>(prefix+std::to_string(found_idx));
	      else
		{
		  int buffer_idx = buffernames.size();
		  std::string buffername = prefix+std::to_string(buffer_idx);
		  buffernames.push_back(buffername);
		  bufferexpressions.emplace_back(this->copy());//just copy everything, no need for further optimisation
		  return std::make_unique<VariableNode<kernelT, evalT>>(buffername);
		}
	    }
	  else
	    return std::make_unique<SqrtNode<kernelT, evalT>>(std::move(this->children_.at(0)->optimize_buffering_constant_terms(buffernames, bufferexpressions, variables, prefix, buffering_cost_threshold)));
	}
      else
	return this->copy();//copies also all children, but fine since we do not want to buffer this
    }
    virtual evalT eval(const std::vector<std::string>& variables, const std::vector<double> & values) const override
    {
      return sqrt(this->children_.at(0)->eval(variables, values));
    }
    virtual std::string draw_node() const override
    {
      std::string result;
      result +=  "sqrt -> { ";
      result += this->children_.at(0)->draw_node() + "}";
      return result;
    }
    //check equality of graphs
    virtual bool operator==(const std::unique_ptr<ComputeGraphNode<kernelT, evalT>>& rhs) const override
    {
      SqrtNode<kernelT, evalT>* rhs_ = dynamic_cast<SqrtNode<kernelT, evalT>*>(rhs.get());
      if (rhs_ != nullptr)
	return this->children_.at(0)->operator==(rhs_->children_.at(0));
      else
	return false;
    }
  };

  
  template<typename kernelT, typename evalT> 
  class SinNode: public ComputeGraphNode<kernelT, evalT> {
  public:
    SinNode(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> child):
      ComputeGraphNode<kernelT, evalT>(std::move(child))
    {
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> copy() const override
    {
      return std::make_unique<SinNode<kernelT, evalT>>(std::move(this->copy_children().at(0)));
    }
    virtual std::string get_kernel() const override
    {      
      return "sin(" + this->children_.at(0)->get_kernel() + ")";
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> diff(std::string variable) const override
    {
      if (!this->variable_in_tree(variable))
	return std::make_unique<ConstantNode<kernelT, evalT>>(0);
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> inner = this->children_.at(0)->diff(variable);
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> arg = this->children_.at(0)->copy();
      std::unique_ptr<CosNode<kernelT, evalT>> outer = std::make_unique<CosNode<kernelT, evalT>>(std::move(arg));
      return std::make_unique<ProdNode<kernelT, evalT>>(make_vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>(std::move(inner), std::move(outer)));
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> substitute(const std::vector<std::string>& variables, const std::vector<double>& values) const override
    {
      return std::make_unique<SinNode<kernelT, evalT>>(std::move(this->children_.at(0)->substitute(variables, values)));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> simplify() const override
    {
      std::unique_ptr child = this->children_.at(0)->simplify();
      if (dynamic_cast<ConstantNode<kernelT, evalT>*>(child.get()) != nullptr)
	{
	  evalT constant = sin(dynamic_cast<ConstantNode<kernelT, evalT>*>(child.get())->number_);
	  return std::make_unique<ConstantNode<kernelT, evalT>>(constant);
	}
      else if (dynamic_cast<AsinNode<kernelT, evalT>*>(child.get()) != nullptr)
	{
	  return dynamic_cast<AsinNode<kernelT, evalT>*>(child.get())->children_.at(0)->copy();
	}
      else
	return std::make_unique<SinNode<kernelT, evalT>>(std::move(child));
    }
    virtual void print(unsigned int level) const override
    {
      std::cout << std::string(level,'\t') << "SinNode with " << this->children_.size() << " children" << std::endl;
      for (unsigned int i=0; i<this->children_.size(); i++)
	this->children_.at(i)->print(level+1);
    }
    virtual double cost() const override
    {
      double c = this->children_.at(0)->cost();
      return 4.0 + c;
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> optimize_buffering_constant_terms(std::vector<std::string>& buffernames, std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>& bufferexpressions, const std::vector<std::string>& variables, const std::string& prefix="morefit_buffer_", float buffering_cost_threshold = 2.0) override
    {
      if (cost() > buffering_cost_threshold)
	{
	  if (!this->contains_other_variables(variables))//expression does not depend on any variables besides the ones given, can optimise!
	    {
	      int found_idx = this->find_in_list(bufferexpressions);
	      if (found_idx >= 0)
		return std::make_unique<VariableNode<kernelT, evalT>>(prefix+std::to_string(found_idx));
	      else
		{
		  int buffer_idx = buffernames.size();
		  std::string buffername = prefix+std::to_string(buffer_idx);
		  buffernames.push_back(buffername);
		  bufferexpressions.emplace_back(this->copy());//just copy everything, no need for further optimisation
		  return std::make_unique<VariableNode<kernelT, evalT>>(buffername);
		}
	    }
	  else
	    return std::make_unique<SinNode<kernelT, evalT>>(std::move(this->children_.at(0)->optimize_buffering_constant_terms(buffernames, bufferexpressions, variables, prefix, buffering_cost_threshold)));
	}
      else
	return this->copy();//copies also all children, but fine since we do not want to buffer this
    }
    virtual evalT eval(const std::vector<std::string>& variables, const std::vector<double> & values) const override
    {
      return sin(this->children_.at(0)->eval(variables, values));
    }
    virtual std::string draw_node() const override
    {
      std::string result;
      result +=  "sin -> { ";
      result += this->children_.at(0)->draw_node() + "}";
      return result;
    }
    //check equality of graphs
    virtual bool operator==(const std::unique_ptr<ComputeGraphNode<kernelT, evalT>>& rhs) const override
    {
      SinNode<kernelT, evalT>* rhs_ = dynamic_cast<SinNode<kernelT, evalT>*>(rhs.get());
      if (rhs_ != nullptr)
	return this->children_.at(0)->operator==(rhs_->children_.at(0));
      else
	return false;
    }
  };


  template<typename kernelT, typename evalT> 
  class AsinNode: public ComputeGraphNode<kernelT, evalT> {
  public:
    AsinNode(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> child):
      ComputeGraphNode<kernelT, evalT>(std::move(child))
    {
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> copy() const override
    {
      return std::make_unique<AsinNode<kernelT, evalT>>(std::move(this->copy_children().at(0)));
    }
    virtual std::string get_kernel() const override
    {      
      return "asin(" + this->children_.at(0)->get_kernel() + ")";
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> diff(std::string variable) const override
    {//1/sqrt(1-x^2)*dx/dvar
      if (!this->variable_in_tree(variable))
	return std::make_unique<ConstantNode<kernelT, evalT>>(0);
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> inner = this->children_.at(0)->diff(variable);
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> arg = this->children_.at(0)->copy();
      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> sumchildren;
      sumchildren.emplace_back(std::make_unique<ConstantNode<kernelT, evalT>>(1.0));
      sumchildren.emplace_back(std::make_unique<NegNode<kernelT, evalT>>(std::make_unique<PowNode<kernelT, evalT>>(std::move(arg), std::make_unique<ConstantNode<kernelT, evalT>>(2.0))));
      std::unique_ptr<InvNode<kernelT, evalT>> outer(std::make_unique<InvNode<kernelT, evalT>>(std::make_unique<SumNode<kernelT, evalT>>(std::move(sumchildren))));
      return std::make_unique<ProdNode<kernelT, evalT>>(make_vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>(std::move(inner), std::move(outer)));
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> substitute(const std::vector<std::string>& variables, const std::vector<double>& values) const override
    {
      return std::make_unique<AsinNode<kernelT, evalT>>(std::move(this->children_.at(0)->substitute(variables, values)));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> simplify() const override
    {
      std::unique_ptr child = this->children_.at(0)->simplify();
      if (dynamic_cast<ConstantNode<kernelT, evalT>*>(child.get()) != nullptr)
	{
	  evalT constant = asin(dynamic_cast<ConstantNode<kernelT, evalT>*>(child.get())->number_);
	  return std::make_unique<ConstantNode<kernelT, evalT>>(constant);
	}
      else if (dynamic_cast<SinNode<kernelT, evalT>*>(child.get()) != nullptr)
	{
	  return dynamic_cast<SinNode<kernelT, evalT>*>(child.get())->children_.at(0)->copy();
	}
      else
	return std::make_unique<AsinNode<kernelT, evalT>>(std::move(child));
    }
    virtual void print(unsigned int level) const override
    {
      std::cout << std::string(level,'\t') << "AsinNode with " << this->children_.size() << " children" << std::endl;
      for (unsigned int i=0; i<this->children_.size(); i++)
	this->children_.at(i)->print(level+1);
    }
    virtual double cost() const override
    {
      double c = this->children_.at(0)->cost();
      return 4.0 + c;
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> optimize_buffering_constant_terms(std::vector<std::string>& buffernames, std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>& bufferexpressions, const std::vector<std::string>& variables, const std::string& prefix="morefit_buffer_", float buffering_cost_threshold = 2.0) override
    {
      if (cost() > buffering_cost_threshold)
	{
	  if (!this->contains_other_variables(variables))//expression does not depend on any variables besides the ones given, can optimise!
	    {
	      int found_idx = this->find_in_list(bufferexpressions);
	      if (found_idx >= 0)
		return std::make_unique<VariableNode<kernelT, evalT>>(prefix+std::to_string(found_idx));
	      else
		{
		  int buffer_idx = buffernames.size();
		  std::string buffername = prefix+std::to_string(buffer_idx);
		  buffernames.push_back(buffername);
		  bufferexpressions.emplace_back(this->copy());//just copy everything, no need for further optimisation
		  return std::make_unique<VariableNode<kernelT, evalT>>(buffername);
		}
	    }
	  else
	    return std::make_unique<AsinNode<kernelT, evalT>>(std::move(this->children_.at(0)->optimize_buffering_constant_terms(buffernames, bufferexpressions, variables, prefix, buffering_cost_threshold)));
	}
      else
	return this->copy();//copies also all children, but fine since we do not want to buffer this
    }
    virtual evalT eval(const std::vector<std::string>& variables, const std::vector<double> & values) const override
    {
      return asin(this->children_.at(0)->eval(variables, values));
    }
    virtual std::string draw_node() const override
    {
      std::string result;
      result +=  "asin -> { ";
      result += this->children_.at(0)->draw_node() + "}";
      return result;
    }
    //check equality of graphs
    virtual bool operator==(const std::unique_ptr<ComputeGraphNode<kernelT, evalT>>& rhs) const override
    {
      AsinNode<kernelT, evalT>* rhs_ = dynamic_cast<AsinNode<kernelT, evalT>*>(rhs.get());
      if (rhs_ != nullptr)
	return this->children_.at(0)->operator==(rhs_->children_.at(0));
      else
	return false;
    }
  };

  template<typename kernelT, typename evalT> 
  class CosNode: public ComputeGraphNode<kernelT, evalT> {
  public:
    CosNode(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> child):
      ComputeGraphNode<kernelT, evalT>(std::move(child))
    {
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> copy() const override
    {
      return std::make_unique<CosNode<kernelT, evalT>>(std::move(this->copy_children().at(0)));
    }
    virtual std::string get_kernel() const override
    {      
      return "cos(" + this->children_.at(0)->get_kernel() + ")";
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> diff(std::string variable) const override
    {
      if (!this->variable_in_tree(variable))
	return std::make_unique<ConstantNode<kernelT, evalT>>(0);
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> inner = this->children_.at(0)->diff(variable);
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> arg = this->children_.at(0)->copy();
      std::unique_ptr<NegNode<kernelT, evalT>> outer = std::make_unique<NegNode<kernelT, evalT>>(std::make_unique<SinNode<kernelT, evalT>>(std::move(arg)));
      return std::make_unique<ProdNode<kernelT, evalT>>(make_vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>(std::move(inner), std::move(outer)));
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> substitute(const std::vector<std::string>& variables, const std::vector<double>& values) const override
    {
      return std::make_unique<CosNode<kernelT, evalT>>(std::move(this->children_.at(0)->substitute(variables, values)));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> simplify() const override
    {
      std::unique_ptr child = this->children_.at(0)->simplify();
      if (dynamic_cast<ConstantNode<kernelT, evalT>*>(child.get()) != nullptr)
	{
	  evalT constant = cos(dynamic_cast<ConstantNode<kernelT, evalT>*>(child.get())->number_);
	  return std::make_unique<ConstantNode<kernelT, evalT>>(constant);
	}
      else if (dynamic_cast<AcosNode<kernelT, evalT>*>(child.get()) != nullptr)
	{
	  return dynamic_cast<AcosNode<kernelT, evalT>*>(child.get())->children_.at(0)->copy();
	}
      else
	return std::make_unique<CosNode<kernelT, evalT>>(std::move(child));
    }
    virtual void print(unsigned int level) const override
    {
      std::cout << std::string(level,'\t') << "CosNode with " << this->children_.size() << " children" << std::endl;
      for (unsigned int i=0; i<this->children_.size(); i++)
	this->children_.at(i)->print(level+1);
    }
    virtual double cost() const override
    {
      double c = this->children_.at(0)->cost();
      return 4.0 + c;
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> optimize_buffering_constant_terms(std::vector<std::string>& buffernames, std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>& bufferexpressions, const std::vector<std::string>& variables, const std::string& prefix="morefit_buffer_", float buffering_cost_threshold = 2.0) override
    {
      if (cost() > buffering_cost_threshold)
	{
	  if (!this->contains_other_variables(variables))//expression does not depend on any variables besides the ones given, can optimise!
	    {
	      int found_idx = this->find_in_list(bufferexpressions);
	      if (found_idx >= 0)
		return std::make_unique<VariableNode<kernelT, evalT>>(prefix+std::to_string(found_idx));
	      else
		{
		  int buffer_idx = buffernames.size();
		  std::string buffername = prefix+std::to_string(buffer_idx);
		  buffernames.push_back(buffername);
		  bufferexpressions.emplace_back(this->copy());//just copy everything, no need for further optimisation
		  return std::make_unique<VariableNode<kernelT, evalT>>(buffername);
		}
	    }
	  else
	    return std::make_unique<CosNode<kernelT, evalT>>(std::move(this->children_.at(0)->optimize_buffering_constant_terms(buffernames, bufferexpressions, variables, prefix, buffering_cost_threshold)));
	}
      else
	return this->copy();//copies also all children, but fine since we do not want to buffer this
    }
    virtual evalT eval(const std::vector<std::string>& variables, const std::vector<double> & values) const override
    {
      return cos(this->children_.at(0)->eval(variables, values));
    }
    virtual std::string draw_node() const override
    {
      std::string result;
      result +=  "cos -> { ";
      result += this->children_.at(0)->draw_node() + "}";
      return result;
    }
    //check equality of graphs
    virtual bool operator==(const std::unique_ptr<ComputeGraphNode<kernelT, evalT>>& rhs) const override
    {
      CosNode<kernelT, evalT>* rhs_ = dynamic_cast<CosNode<kernelT, evalT>*>(rhs.get());
      if (rhs_ != nullptr)
	return this->children_.at(0)->operator==(rhs_->children_.at(0));
      else
	return false;
    }
  };


  template<typename kernelT, typename evalT> 
  class AcosNode: public ComputeGraphNode<kernelT, evalT> {
  public:
    AcosNode(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> child):
      ComputeGraphNode<kernelT, evalT>(std::move(child))
    {
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> copy() const override
    {
      return std::make_unique<AcosNode<kernelT, evalT>>(std::move(this->copy_children().at(0)));
    }
    virtual std::string get_kernel() const override
    {      
      return "acos(" + this->children_.at(0)->get_kernel() + ")";
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> diff(std::string variable) const override
    {//-1/sqrt(1-x^2)*dx/dvar
      if (!this->variable_in_tree(variable))
	return std::make_unique<ConstantNode<kernelT, evalT>>(0);
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> inner = this->children_.at(0)->diff(variable);
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> arg = this->children_.at(0)->copy();
      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> sumchildren;
      sumchildren.emplace_back(std::make_unique<ConstantNode<kernelT, evalT>>(1.0));
      sumchildren.emplace_back(std::make_unique<NegNode<kernelT, evalT>>(std::make_unique<PowNode<kernelT, evalT>>(std::move(arg), std::make_unique<ConstantNode<kernelT, evalT>>(2.0))));
      std::unique_ptr<NegNode<kernelT, evalT>> outer(std::make_unique<NegNode<kernelT, evalT>>(std::make_unique<InvNode<kernelT, evalT>>(std::make_unique<SumNode<kernelT, evalT>>(std::move(sumchildren)))));      
      return std::make_unique<ProdNode<kernelT, evalT>>(make_vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>(std::move(inner), std::move(outer)));
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> substitute(const std::vector<std::string>& variables, const std::vector<double>& values) const override
    {
      return std::make_unique<AcosNode<kernelT, evalT>>(std::move(this->children_.at(0)->substitute(variables, values)));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> simplify() const override
    {
      std::unique_ptr child = this->children_.at(0)->simplify();
      if (dynamic_cast<ConstantNode<kernelT, evalT>*>(child.get()) != nullptr)
	{
	  evalT constant = acos(dynamic_cast<ConstantNode<kernelT, evalT>*>(child.get())->number_);
	  return std::make_unique<ConstantNode<kernelT, evalT>>(constant);
	}
      else if (dynamic_cast<CosNode<kernelT, evalT>*>(child.get()) != nullptr)
	{
	  return dynamic_cast<CosNode<kernelT, evalT>*>(child.get())->children_.at(0)->copy();
	}
      else
	return std::make_unique<AcosNode<kernelT, evalT>>(std::move(child));
    }
    virtual void print(unsigned int level) const override
    {
      std::cout << std::string(level,'\t') << "AcosNode with " << this->children_.size() << " children" << std::endl;
      for (unsigned int i=0; i<this->children_.size(); i++)
	this->children_.at(i)->print(level+1);
    }
    virtual double cost() const override
    {
      double c = this->children_.at(0)->cost();
      return 4.0 + c;
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> optimize_buffering_constant_terms(std::vector<std::string>& buffernames, std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>& bufferexpressions, const std::vector<std::string>& variables, const std::string& prefix="morefit_buffer_", float buffering_cost_threshold = 2.0) override
    {
      if (cost() > buffering_cost_threshold)
	{
	  if (!this->contains_other_variables(variables))//expression does not depend on any variables besides the ones given, can optimise!
	    {
	      int found_idx = this->find_in_list(bufferexpressions);
	      if (found_idx >= 0)
		return std::make_unique<VariableNode<kernelT, evalT>>(prefix+std::to_string(found_idx));
	      else
		{
		  int buffer_idx = buffernames.size();
		  std::string buffername = prefix+std::to_string(buffer_idx);
		  buffernames.push_back(buffername);
		  bufferexpressions.emplace_back(this->copy());//just copy everything, no need for further optimisation
		  return std::make_unique<VariableNode<kernelT, evalT>>(buffername);
		}
	    }
	  else
	    return std::make_unique<AcosNode<kernelT, evalT>>(std::move(this->children_.at(0)->optimize_buffering_constant_terms(buffernames, bufferexpressions, variables, prefix, buffering_cost_threshold)));
	}
      else
	return this->copy();//copies also all children, but fine since we do not want to buffer this
    }
    virtual evalT eval(const std::vector<std::string>& variables, const std::vector<double> & values) const override
    {
      return acos(this->children_.at(0)->eval(variables, values));
    }
    virtual std::string draw_node() const override
    {
      std::string result;
      result +=  "acos -> { ";
      result += this->children_.at(0)->draw_node() + "}";
      return result;
    }
    //check equality of graphs
    virtual bool operator==(const std::unique_ptr<ComputeGraphNode<kernelT, evalT>>& rhs) const override
    {
      AcosNode<kernelT, evalT>* rhs_ = dynamic_cast<AcosNode<kernelT, evalT>*>(rhs.get());
      if (rhs_ != nullptr)
	return this->children_.at(0)->operator==(rhs_->children_.at(0));
      else
	return false;
    }
  };




  template<typename kernelT, typename evalT> 
  class TanNode: public ComputeGraphNode<kernelT, evalT> {
  public:
    TanNode(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> child):
      ComputeGraphNode<kernelT, evalT>(std::move(child))
    {
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> copy() const override
    {
      return std::make_unique<TanNode<kernelT, evalT>>(std::move(this->copy_children().at(0)));
    }
    virtual std::string get_kernel() const override
    {      
      return "tan(" + this->children_.at(0)->get_kernel() + ")";
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> diff(std::string variable) const override
    {
      if (!this->variable_in_tree(variable))
	return std::make_unique<ConstantNode<kernelT, evalT>>(0);
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> inner = this->children_.at(0)->diff(variable);
      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> sum_children;
      sum_children.emplace_back(std::make_unique<ConstantNode<kernelT, evalT>>(1.0));
      sum_children.emplace_back(std::make_unique<PowNode<kernelT, evalT>>(this->children_.at(0)->copy(), std::make_unique<ConstantNode<kernelT, evalT>>(2.0)));
      std::unique_ptr<SumNode<kernelT, evalT>> outer(std::make_unique<SumNode<kernelT, evalT>>(std::move(sum_children)));
      return std::make_unique<ProdNode<kernelT, evalT>>(make_vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>(std::move(inner), std::move(outer)));
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> substitute(const std::vector<std::string>& variables, const std::vector<double>& values) const override
    {
      return std::make_unique<TanNode<kernelT, evalT>>(std::move(this->children_.at(0)->substitute(variables, values)));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> simplify() const override
    {
      std::unique_ptr child = this->children_.at(0)->simplify();
      if (dynamic_cast<ConstantNode<kernelT, evalT>*>(child.get()) != nullptr)
	{
	  evalT constant = tan(dynamic_cast<ConstantNode<kernelT, evalT>*>(child.get())->number_);
	  return std::make_unique<ConstantNode<kernelT, evalT>>(constant);
	}
      else if (dynamic_cast<AtanNode<kernelT, evalT>*>(child.get()) != nullptr)
	{
	  return dynamic_cast<AtanNode<kernelT, evalT>*>(child.get())->children_.at(0)->copy();
	}
      else
	return std::make_unique<TanNode<kernelT, evalT>>(std::move(child));
    }
    virtual void print(unsigned int level) const override
    {
      std::cout << std::string(level,'\t') << "TanNode with " << this->children_.size() << " children" << std::endl;
      for (unsigned int i=0; i<this->children_.size(); i++)
	this->children_.at(i)->print(level+1);
    }
    virtual double cost() const override
    {
      double c = this->children_.at(0)->cost();
      return 4.0 + c;
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> optimize_buffering_constant_terms(std::vector<std::string>& buffernames, std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>& bufferexpressions, const std::vector<std::string>& variables, const std::string& prefix="morefit_buffer_", float buffering_cost_threshold = 2.0) override
    {
      if (cost() > buffering_cost_threshold)
	{
	  if (!this->contains_other_variables(variables))//expression does not depend on any variables besides the ones given, can optimise!
	    {
	      int found_idx = this->find_in_list(bufferexpressions);
	      if (found_idx >= 0)
		return std::make_unique<VariableNode<kernelT, evalT>>(prefix+std::to_string(found_idx));
	      else
		{
		  int buffer_idx = buffernames.size();
		  std::string buffername = prefix+std::to_string(buffer_idx);
		  buffernames.push_back(buffername);
		  bufferexpressions.emplace_back(this->copy());//just copy everything, no need for further optimisation
		  return std::make_unique<VariableNode<kernelT, evalT>>(buffername);
		}
	    }
	  else
	    return std::make_unique<TanNode<kernelT, evalT>>(std::move(this->children_.at(0)->optimize_buffering_constant_terms(buffernames, bufferexpressions, variables, prefix, buffering_cost_threshold)));
	}
      else
	return this->copy();//copies also all children, but fine since we do not want to buffer this
    }
    virtual evalT eval(const std::vector<std::string>& variables, const std::vector<double> & values) const override
    {
      return tan(this->children_.at(0)->eval(variables, values));
    }
    virtual std::string draw_node() const override
    {
      std::string result;
      result +=  "tan -> { ";
      result += this->children_.at(0)->draw_node() + "}";
      return result;
    }
    //check equality of graphs
    virtual bool operator==(const std::unique_ptr<ComputeGraphNode<kernelT, evalT>>& rhs) const override
    {
      TanNode<kernelT, evalT>* rhs_ = dynamic_cast<TanNode<kernelT, evalT>*>(rhs.get());
      if (rhs_ != nullptr)
	return this->children_.at(0)->operator==(rhs_->children_.at(0));
      else
	return false;
    }
  };


  template<typename kernelT, typename evalT> 
  class AtanNode: public ComputeGraphNode<kernelT, evalT> {
  public:
    AtanNode(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> child):
      ComputeGraphNode<kernelT, evalT>(std::move(child))
    {
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> copy() const override
    {
      return std::make_unique<AtanNode<kernelT, evalT>>(std::move(this->copy_children().at(0)));
    }
    virtual std::string get_kernel() const override
    {      
      return "atan(" + this->children_.at(0)->get_kernel() + ")";
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> diff(std::string variable) const override
    {//1/sqrt(1+x^2)*dx/dvar
      if (!this->variable_in_tree(variable))
	return std::make_unique<ConstantNode<kernelT, evalT>>(0);
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> inner = this->children_.at(0)->diff(variable);
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> arg = this->children_.at(0)->copy();
      std::unique_ptr<InvNode<kernelT, evalT>> outer(std::make_unique<InvNode<kernelT, evalT>>(std::make_unique<SumNode<kernelT, evalT>>(std::make_unique<ConstantNode<kernelT, evalT>>(1.0),std::make_unique<PowNode<kernelT, evalT>>(std::move(arg),std::make_unique<ConstantNode<kernelT, evalT>>(2.0)))));
      return std::make_unique<ProdNode<kernelT, evalT>>(make_vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>(std::move(inner), std::move(outer)));
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> substitute(const std::vector<std::string>& variables, const std::vector<double>& values) const override
    {
      return std::make_unique<AtanNode<kernelT, evalT>>(std::move(this->children_.at(0)->substitute(variables, values)));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> simplify() const override
    {
      std::unique_ptr child = this->children_.at(0)->simplify();
      if (dynamic_cast<ConstantNode<kernelT, evalT>*>(child.get()) != nullptr)
	{
	  evalT constant = atan(dynamic_cast<ConstantNode<kernelT, evalT>*>(child.get())->number_);
	  return std::make_unique<ConstantNode<kernelT, evalT>>(constant);
	}
      else if (dynamic_cast<TanNode<kernelT, evalT>*>(child.get()) != nullptr)
	{
	  return dynamic_cast<TanNode<kernelT, evalT>*>(child.get())->children_.at(0)->copy();
	}
      else
	return std::make_unique<AtanNode<kernelT, evalT>>(std::move(child));
    }
    virtual void print(unsigned int level) const override
    {
      std::cout << std::string(level,'\t') << "AtanNode with " << this->children_.size() << " children" << std::endl;
      for (unsigned int i=0; i<this->children_.size(); i++)
	this->children_.at(i)->print(level+1);
    }
    virtual double cost() const override
    {
      double c = this->children_.at(0)->cost();
      return 4.0 + c;
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> optimize_buffering_constant_terms(std::vector<std::string>& buffernames, std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>& bufferexpressions, const std::vector<std::string>& variables, const std::string& prefix="morefit_buffer_", float buffering_cost_threshold = 2.0) override
    {
      if (cost() > buffering_cost_threshold)
	{
	  if (!this->contains_other_variables(variables))//expression does not depend on any variables besides the ones given, can optimise!
	    {
	      int found_idx = this->find_in_list(bufferexpressions);
	      if (found_idx >= 0)
		return std::make_unique<VariableNode<kernelT, evalT>>(prefix+std::to_string(found_idx));
	      else
		{
		  int buffer_idx = buffernames.size();
		  std::string buffername = prefix+std::to_string(buffer_idx);
		  buffernames.push_back(buffername);
		  bufferexpressions.emplace_back(this->copy());//just copy everything, no need for further optimisation
		  return std::make_unique<VariableNode<kernelT, evalT>>(buffername);
		}
	    }
	  else
	    return std::make_unique<AtanNode<kernelT, evalT>>(std::move(this->children_.at(0)->optimize_buffering_constant_terms(buffernames, bufferexpressions, variables, prefix, buffering_cost_threshold)));
	}
      else
	return this->copy();//copies also all children, but fine since we do not want to buffer this
    }
    virtual evalT eval(const std::vector<std::string>& variables, const std::vector<double> & values) const override
    {
      return atan(this->children_.at(0)->eval(variables, values));
    }
    virtual std::string draw_node() const override
    {
      std::string result;
      result +=  "atan -> { ";
      result += this->children_.at(0)->draw_node() + "}";
      return result;
    }
    //check equality of graphs
    virtual bool operator==(const std::unique_ptr<ComputeGraphNode<kernelT, evalT>>& rhs) const override
    {
      AtanNode<kernelT, evalT>* rhs_ = dynamic_cast<AtanNode<kernelT, evalT>*>(rhs.get());
      if (rhs_ != nullptr)
	return this->children_.at(0)->operator==(rhs_->children_.at(0));
      else
	return false;
    }
  };

  template<typename kernelT, typename evalT> 
  class ErfNode: public ComputeGraphNode<kernelT, evalT> {
  public:
    ErfNode(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> child):
      ComputeGraphNode<kernelT, evalT>(std::move(child))
    {
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> copy() const override
    {
      return std::make_unique<ErfNode<kernelT, evalT>>(std::move(this->copy_children().at(0)));
    }
    virtual std::string get_kernel() const override
    {      
      return "erf(" + this->children_.at(0)->get_kernel() + ")";
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> diff(std::string variable) const override
    {
      if (!this->variable_in_tree(variable))
	return std::make_unique<ConstantNode<kernelT, evalT>>(0);
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> inner = this->children_.at(0)->diff(variable);
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> arg = this->children_.at(0)->copy();
      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> prod_children;
      prod_children.emplace_back(std::make_unique<ConstantNode<kernelT, evalT>>(2.0));
      prod_children.emplace_back(std::make_unique<InvNode<kernelT, evalT>>(std::make_unique<SqrtNode<kernelT, evalT>>(std::make_unique<ConstantNode<kernelT, evalT>>(M_PI))));
      prod_children.emplace_back(std::make_unique<ExpNode<kernelT, evalT>>(std::make_unique<NegNode<kernelT, evalT>>(std::make_unique<PowNode<kernelT, evalT>>(std::move(arg),std::make_unique<ConstantNode<kernelT, evalT>>(2.0)))));//TODO instead z*z?
      std::unique_ptr<ProdNode<kernelT, evalT>> outer(std::make_unique<ProdNode<kernelT, evalT>>(std::move(prod_children)));
      return std::make_unique<ProdNode<kernelT, evalT>>(make_vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>(std::move(inner), std::move(outer)));
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> substitute(const std::vector<std::string>& variables, const std::vector<double>& values) const override
    {
      return std::make_unique<ErfNode<kernelT, evalT>>(std::move(this->children_.at(0)->substitute(variables, values)));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> simplify() const override
    {
      std::unique_ptr child = this->children_.at(0)->simplify();
      if (dynamic_cast<ConstantNode<kernelT, evalT>*>(child.get()) != nullptr)
	{
	  evalT constant = erf(dynamic_cast<ConstantNode<kernelT, evalT>*>(child.get())->number_);
	  return std::make_unique<ConstantNode<kernelT, evalT>>(constant);
	}
      else
	return std::make_unique<ErfNode<kernelT, evalT>>(std::move(child));
    }
    virtual void print(unsigned int level) const override
    {
      std::cout << std::string(level,'\t') << "ErfNode with " << this->children_.size() << " children" << std::endl;
      for (unsigned int i=0; i<this->children_.size(); i++)
	this->children_.at(i)->print(level+1);
    }
    virtual double cost() const override
    {
      double c = this->children_.at(0)->cost();
      return 4.0 + c;
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> optimize_buffering_constant_terms(std::vector<std::string>& buffernames, std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>& bufferexpressions, const std::vector<std::string>& variables, const std::string& prefix="morefit_buffer_", float buffering_cost_threshold = 2.0) override
    {
      if (cost() > buffering_cost_threshold)
	{
	  if (!this->contains_other_variables(variables))//expression does not depend on any variables besides the ones given, can optimise!
	    {
	      int found_idx = this->find_in_list(bufferexpressions);
	      if (found_idx >= 0)
		return std::make_unique<VariableNode<kernelT, evalT>>(prefix+std::to_string(found_idx));
	      else
		{
		  int buffer_idx = buffernames.size();
		  std::string buffername = prefix+std::to_string(buffer_idx);
		  buffernames.push_back(buffername);
		  bufferexpressions.emplace_back(this->copy());//just copy everything, no need for further optimisation
		  return std::make_unique<VariableNode<kernelT, evalT>>(buffername);
		}
	    }
	  else
	    return std::make_unique<ErfNode<kernelT, evalT>>(std::move(this->children_.at(0)->optimize_buffering_constant_terms(buffernames, bufferexpressions, variables, prefix, buffering_cost_threshold)));
	}
      else
	return this->copy();//copies also all children, but fine since we do not want to buffer this
    }
    virtual evalT eval(const std::vector<std::string>& variables, const std::vector<double> & values) const override
    {
      return erf(this->children_.at(0)->eval(variables, values));
    }
    virtual std::string draw_node() const override
    {
      std::string result;
      result +=  "erf -> { ";
      result += this->children_.at(0)->draw_node() + "}";
      return result;
    }
    //check equality of graphs
    virtual bool operator==(const std::unique_ptr<ComputeGraphNode<kernelT, evalT>>& rhs) const override
    {
      ErfNode<kernelT, evalT>* rhs_ = dynamic_cast<ErfNode<kernelT, evalT>*>(rhs.get());
      if (rhs_ != nullptr)
	return this->children_.at(0)->operator==(rhs_->children_.at(0));
      else
	return false;
    }
  };
  
  //a variable, can be either a dimension or a parameter
  template<typename kernelT, typename evalT> 
  class VariableNode: public ComputeGraphNode<kernelT, evalT> {
  public:
    std::string name_;
    VariableNode(std::string name) 
      : name_(name)
    {
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> copy() const override
    {
      return std::make_unique<VariableNode<kernelT, evalT>>(name_);
    }
    //renames single variable, does change this expression (ie. is non-const)
    virtual void rename_variable(const std::string& variable, const std::string& new_name)
    {
      if (name_ == variable)
	name_ = new_name;
    }
    virtual std::string get_kernel() const override
    {
      return name_;
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> diff(std::string variable) const override
    {
      if (variable == name_)
	return std::make_unique<ConstantNode<kernelT, evalT>>(1);
      else
	return std::make_unique<ConstantNode<kernelT, evalT>>(0);
    }    
    //substitute named variables (can be dimensions/parameters) with constants
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> substitute(const std::vector<std::string>& variables, const std::vector<double>& values) const override
    {
      auto it = std::find(variables.begin(), variables.end(), name_);
      if (it == variables.end())
	return this->copy();
      else
	{
	  unsigned int var_idx = std::distance(variables.begin(), it);
	  return std::make_unique<ConstantNode<kernelT, evalT>>(values.at(var_idx));
	}
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> simplify() const override
    {
      return std::make_unique<VariableNode<kernelT, evalT>>(name_);
    }
    virtual void print(unsigned int level) const override
    {
      std::cout << std::string(level,'\t') << "VariableNode: " << name_ << std::endl;
    }
    virtual bool variable_in_tree(const std::string& variable) const override
    {
      if (variable == name_)
	return true;
      else
	return false;
    }
    virtual bool contains_other_variables(const std::vector<std::string>& variables) const override
    {
      if (std::find(variables.begin(), variables.end(), name_) == variables.end())//did not find this variable in list of given variables
	return true;
      else
	return false;
    }
    virtual double cost() const override
    {
      return 1.0;
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> optimize_buffering_constant_terms(std::vector<std::string>& buffernames, std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>& bufferexpressions, const std::vector<std::string>& variables, const std::string& prefix="morefit_buffer_", float buffering_cost_threshold = 2.0) override
    {
      return std::make_unique<VariableNode<kernelT, evalT>>(name_);
    }
    virtual evalT eval(const std::vector<std::string>& variables, const std::vector<double> & values) const override
    {
      auto it = std::find(variables.begin(), variables.end(), name_);
      if (it == variables.end())
	return std::numeric_limits<evalT>::quiet_NaN();
      else
	{
	  unsigned int var_idx = std::distance(variables.begin(), it);
	  return values.at(var_idx);
	}
    }
    virtual std::string draw_node() const override
    {
      return name_;
    }
    //check equality of graphs
    virtual bool operator==(const std::unique_ptr<ComputeGraphNode<kernelT, evalT>>& rhs) const override
    {
      VariableNode<kernelT, evalT>* rhs_ = dynamic_cast<VariableNode<kernelT, evalT>*>(rhs.get());
      if (rhs_ != nullptr)
	return this->name_ == rhs_->name_;
      else
	return false;
    }
  };


  //ternary operation
  template<typename kernelT, typename evalT> 
  class ConditionalNode: public ComputeGraphNode<kernelT, evalT> {
  public:
    enum comparator_type {Equal, Unequal, Larger, Smaller, LargerEqual, SmallerEqual};//All comparisons with zero
  private:
    comparator_type comparator_;
  public:
    ConditionalNode(comparator_type comparator, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> expression, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> childa, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> childb):
      comparator_(comparator)
    {
      this->children_.emplace_back(std::move(expression));
      this->children_.emplace_back(std::move(childa));
      this->children_.emplace_back(std::move(childb));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> copy() const override
    {
      return std::make_unique<ConditionalNode<kernelT, evalT>>(this->comparator_,  std::move(this->children_.at(0)->copy()), std::move(this->children_.at(1)->copy()), std::move(this->children_.at(2)->copy()));
    }
    virtual std::string get_kernel() const override
    {
      std::string result = "((";
      result += this->children_.at(0)->get_kernel();
      switch (comparator_) {
      case comparator_type::Equal: result += "==0.0) ? "; break;
      case comparator_type::Unequal: result += "!=0.0) ? "; break;
      case comparator_type::Larger: result += ">0.0) ? "; break;
      case comparator_type::Smaller: result += "<0.0) ? "; break;
      case comparator_type::LargerEqual: result += ">=0.0) ? "; break;
      case comparator_type::SmallerEqual: result += "<=0.0) ? "; break;
      default:
	std::cout << "Unknown comparator type" << std::endl;
	assert(0);
      }
      result += this->children_.at(1)->get_kernel();
      result += " : ";
      result += this->children_.at(2)->get_kernel();
      result += ")";
      return result;
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> diff(std::string variable) const override
    {
      //variable is neither in child 1 nor in child 2
      if (!this->children_.at(1)->variable_in_tree(variable) && !this->children_.at(2)->variable_in_tree(variable))
	return std::make_unique<ConstantNode<kernelT, evalT>>(0);
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> childa = std::move(this->children_.at(1)->diff(variable));
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> childb = std::move(this->children_.at(2)->diff(variable));
      return std::make_unique<ConditionalNode<kernelT, evalT>>(this->comparator_,  std::move(this->children_.at(0)->copy()), std::move(childa), std::move(childb));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> substitute(const std::vector<std::string>& variables, const std::vector<double>& values) const override
    {
      //substitutes both children and expression
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> expression = std::move(this->children_.at(0)->substitute(variables, values));
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> childa = std::move(this->children_.at(1)->substitute(variables, values));
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> childb = std::move(this->children_.at(2)->substitute(variables, values));
      return std::make_unique<ConditionalNode<kernelT, evalT>>(this->comparator_,  std::move(expression), std::move(childa), std::move(childb));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> simplify() const override
    {
      //simplify expression first
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> expression = std::move(this->children_.at(0)->simplify());
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> childa = std::move(this->children_.at(1)->simplify());
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> childb = std::move(this->children_.at(2)->simplify());
      //if resulting expression constant can only return one child
      if (dynamic_cast<ConstantNode<kernelT, evalT>*>(expression.get()) != nullptr)
	{
	  evalT constant = dynamic_cast<ConstantNode<kernelT, evalT>*>(expression.get())->number_;
	  switch (comparator_) {
	  case comparator_type::Equal:
	    return (constant == 0.0 ? std::move(childa) : std::move(childb));
	    break;
	  case comparator_type::Unequal:
	    return (constant != 0.0 ? std::move(childa) : std::move(childb));
	    break;
	  case comparator_type::Larger:
	    return (constant > 0.0 ? std::move(childa) : std::move(childb));
	    break;
	  case comparator_type::Smaller:
	    return (constant < 0.0 ? std::move(childa) : std::move(childb));
	    break;
	  case comparator_type::LargerEqual:
	    return (constant >= 0.0 ? std::move(childa) : std::move(childb));
	    break;
	  case comparator_type::SmallerEqual:
	    return (constant <= 0.0 ? std::move(childa) : std::move(childb));
	    break;
	  default:
	    std::cout << "Unknown comparator type" << std::endl;
	    assert(0);
	  }
	}
      else
	return std::make_unique<ConditionalNode<kernelT, evalT>>(this->comparator_,  std::move(expression), std::move(childa), std::move(childb));
    }
    virtual void print(unsigned int level) const override
    {
      std::cout << std::string(level,'\t') << "ConditionalNode checking expression";
      switch (comparator_) {
      case comparator_type::Equal:
	std::cout << "==0";
	break;
      case comparator_type::Unequal:
	std::cout << "!=0";
	break;
      case comparator_type::Larger:
	std::cout << ">0";
	break;
      case comparator_type::Smaller:
	std::cout << "<0";
	break;
      case comparator_type::LargerEqual:
	std::cout << ">=0";
	break;
      case comparator_type::SmallerEqual:
	std::cout << "<=0";
	break;
      default:
	std::cout << "Unknown comparator type" << std::endl;
	assert(0);
      }      
      std::cout << " with " << this->children_.size() << " children" << std::endl;
      for (unsigned int i=0; i<this->children_.size(); i++)
	this->children_.at(i)->print(level+1);
    }
    virtual double cost() const override
    {
      double c = 10.0;
      for (unsigned int i=0; i<this->children_.size(); i++)
	c += this->children_.at(i)->cost();
      return c;
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> optimize_buffering_constant_terms(std::vector<std::string>& buffernames, std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>& bufferexpressions, const std::vector<std::string>& variables, const std::string& prefix="morefit_buffer_", float buffering_cost_threshold = 2.0) override
    {
      if (cost() > buffering_cost_threshold)
	{
	  if (!this->contains_other_variables(variables))//expression does not depend on any variables besides the ones given, can optimise!
	    {
	      int found_idx = this->find_in_list(bufferexpressions);
	      if (found_idx >= 0)
		return std::make_unique<VariableNode<kernelT, evalT>>(prefix+std::to_string(found_idx));
	      else
		{	      
		  int buffer_idx = buffernames.size();
		  std::string buffername = prefix+std::to_string(buffer_idx);
		  buffernames.push_back(buffername);
		  bufferexpressions.emplace_back(this->copy());//just copy everything, no need for further optimisation
		  return std::make_unique<VariableNode<kernelT, evalT>>(buffername);
		}
	    }
	  else
	    {
	      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> expression = std::move(this->children_.at(0)->optimize_buffering_constant_terms(buffernames, bufferexpressions, variables, prefix, buffering_cost_threshold));
	      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> childa = std::move(this->children_.at(1)->optimize_buffering_constant_terms(buffernames, bufferexpressions, variables, prefix, buffering_cost_threshold));
	      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> childb = std::move(this->children_.at(2)->optimize_buffering_constant_terms(buffernames, bufferexpressions, variables, prefix, buffering_cost_threshold));
	      return std::make_unique<ConditionalNode<kernelT, evalT>>(this->comparator_,  std::move(expression), std::move(childa), std::move(childb));
	    }
	}
      else
	return this->copy();//copies also all children, but fine since we do not want to buffer this
    }
    virtual evalT eval(const std::vector<std::string>& variables, const std::vector<double> & values) const override
    {
      evalT expression = this->children_.at(0)->eval(variables, values);
      switch (comparator_) {
      case comparator_type::Equal:
	return (expression == 0.0 ? this->children_.at(1)->eval(variables, values) : this->children_.at(2)->eval(variables, values));
	break;
      case comparator_type::Unequal:
	return (expression != 0.0 ? this->children_.at(1)->eval(variables, values) : this->children_.at(2)->eval(variables, values));
	break;
      case comparator_type::Larger:
	return (expression > 0.0 ? this->children_.at(1)->eval(variables, values) : this->children_.at(2)->eval(variables, values));
	break;
      case comparator_type::Smaller:
	return (expression < 0.0 ? this->children_.at(1)->eval(variables, values) : this->children_.at(2)->eval(variables, values));
	break;
      case comparator_type::LargerEqual:
	return (expression >= 0.0 ? this->children_.at(1)->eval(variables, values) : this->children_.at(2)->eval(variables, values));
	break;
      case comparator_type::SmallerEqual:
	return (expression <= 0.0 ? this->children_.at(1)->eval(variables, values) : this->children_.at(2)->eval(variables, values));
	break;
      default:
	std::cout << "Unknown comparator type" << std::endl;
	assert(0);
	return -1.0;
      }
    }
    virtual std::string draw_node() const override
    {
      std::string result("\"");
      result += this->children_.at(0)->get_kernel();
      switch (comparator_) {
      case comparator_type::Equal:
	result +=  "==0?\" -> { ";
	break;
      case comparator_type::Unequal:
	result +=  "!=0?\" -> { ";
	break;
      case comparator_type::Larger:
	result +=  ">0?\" -> { ";
	break;
      case comparator_type::Smaller:
	result +=  "<0?\" -> { ";
	break;
      case comparator_type::LargerEqual:
	result +=  ">=0?\" -> { ";
	break;
      case comparator_type::SmallerEqual:
	result +=  "<=0?\" -> { ";
	break;
      default:
	std::cout << "Unknown comparator type" << std::endl;
	assert(0);
      }
      for (unsigned int i=1; i<this->children_.size(); i++)
	result += this->children_.at(i)->draw_node() + (i == this->children_.size()-1 ? " }": ", ");
      return result;
    }
    //check equality of graphs
    virtual bool operator==(const std::unique_ptr<ComputeGraphNode<kernelT, evalT>>& rhs) const override
    {
      ConditionalNode<kernelT, evalT>* rhs_ = dynamic_cast<ConditionalNode<kernelT, evalT>*>(rhs.get());
      if (rhs_ != nullptr)
	{
	  if (this->children_.size() != rhs_->children_.size())
	    return false;
	  for (unsigned int i=0; i<this->children_.size(); i++)
	    {
	      if (!this->children_.at(i)->operator==(rhs_->children_.at(i)))
		return false;
	    }
	  return true;
	}
      else
	return false;
    }
  };

  
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Variable(std::string name)
  {
    return std::make_unique<VariableNode<kernelT, evalT>>(name);
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Constant(evalT c)
  {
    return std::make_unique<ConstantNode<kernelT, evalT>>(c);
  };

  //operator==
  template<typename kernelT, typename evalT>
  bool operator==(const std::unique_ptr<ComputeGraphNode<kernelT, evalT>>& lhs, const std::unique_ptr<ComputeGraphNode<kernelT, evalT>>& rhs)
  {
    return lhs->operator==(rhs);
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> ConditionalEqual(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> expression, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> B)
  {
    return std::make_unique<ConditionalNode<kernelT, evalT>>(ConditionalNode<kernelT, evalT>::comparator_type::Equal, std::move(expression), std::move(A), std::move(B));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> ConditionalUnequal(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> expression, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> B)
  {
    return std::make_unique<ConditionalNode<kernelT, evalT>>(ConditionalNode<kernelT, evalT>::comparator_type::Unequal, std::move(expression), std::move(A), std::move(B));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> ConditionalLarger(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> expression, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> B)
  {
    return std::make_unique<ConditionalNode<kernelT, evalT>>(ConditionalNode<kernelT, evalT>::comparator_type::Larger, std::move(expression), std::move(A), std::move(B));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> ConditionalLargerEqual(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> expression, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> B)
  {
    return std::make_unique<ConditionalNode<kernelT, evalT>>(ConditionalNode<kernelT, evalT>::comparator_type::LargerEqual, std::move(expression), std::move(A), std::move(B));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> ConditionalSmaller(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> expression, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> B)
  {
    return std::make_unique<ConditionalNode<kernelT, evalT>>(ConditionalNode<kernelT, evalT>::comparator_type::Smaller, std::move(expression), std::move(A), std::move(B));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> ConditionalSmallerEqual(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> expression, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> B)
  {
    return std::make_unique<ConditionalNode<kernelT, evalT>>(ConditionalNode<kernelT, evalT>::comparator_type::SmallerEqual, std::move(expression), std::move(A), std::move(B));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Sum(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> B)
  {
    return std::make_unique<SumNode<kernelT, evalT>>(std::move(A), std::move(B));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Prod(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> B)
  {
    return std::make_unique<ProdNode<kernelT, evalT>>(std::move(A), std::move(B));
  };

  //operator+
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator+(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> B)
  {
    return std::make_unique<SumNode<kernelT, evalT>>(std::move(A), std::move(B));
  };

  /*
  //can be problematic when compute graph is not wanted
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator+(std::string A, std::string B)
  {
    return std::make_unique<SumNode<kernelT, evalT>>(std::move(std::make_unique<VariableNode<kernelT, evalT>>(A)), std::move(std::make_unique<VariableNode<kernelT, evalT>>(B)));
  };
  */  

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator+(evalT A, std::string B)
  {
    return std::make_unique<SumNode<kernelT, evalT>>(std::move(std::make_unique<ConstantNode<kernelT, evalT>>(A)), std::move(std::make_unique<VariableNode<kernelT, evalT>>(B)));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator+(std::string A, evalT B)
  {
    return std::make_unique<SumNode<kernelT, evalT>>(std::move(std::make_unique<VariableNode<kernelT, evalT>>(A)), std::move(std::make_unique<ConstantNode<kernelT, evalT>>(B)));
  };
  
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator+(evalT A, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> B)
  {
    return std::make_unique<SumNode<kernelT, evalT>>(std::move(std::make_unique<ConstantNode<kernelT, evalT>>(A)), std::move(B));
  };
  
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator+(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A, evalT B)
  {
    return std::make_unique<SumNode<kernelT, evalT>>(std::move(A), std::move(std::make_unique<ConstantNode<kernelT, evalT>>(B)));
  };
    
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator+(std::string A, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> B)
  {
    return std::make_unique<SumNode<kernelT, evalT>>(std::move(std::make_unique<VariableNode<kernelT, evalT>>(A)), std::move(B));
  };
  
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator+(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A, std::string B)
  {
    return std::make_unique<SumNode<kernelT, evalT>>(std::move(A), std::move(std::make_unique<VariableNode<kernelT, evalT>>(B)));
  };

  //operator*
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator*(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> B)
  {
    return std::make_unique<ProdNode<kernelT, evalT>>(std::move(A), std::move(B));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator*(std::string A, std::string B)
  {
    return std::make_unique<ProdNode<kernelT, evalT>>(std::move(std::make_unique<VariableNode<kernelT, evalT>>(A)), std::move(std::make_unique<VariableNode<kernelT, evalT>>(B)));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator*(evalT A, std::string B)
  {
    return std::make_unique<ProdNode<kernelT, evalT>>(std::move(std::make_unique<ConstantNode<kernelT, evalT>>(A)), std::move(std::make_unique<VariableNode<kernelT, evalT>>(B)));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator*(std::string A, evalT B)
  {
    return std::make_unique<ProdNode<kernelT, evalT>>(std::move(std::make_unique<VariableNode<kernelT, evalT>>(A)), std::move(std::make_unique<ConstantNode<kernelT, evalT>>(B)));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator*(evalT A, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> B)
  {
    return std::make_unique<ProdNode<kernelT, evalT>>(std::move(std::make_unique<ConstantNode<kernelT, evalT>>(A)), std::move(B));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator*(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A, evalT B)
  {
    return std::make_unique<ProdNode<kernelT, evalT>>(std::move(A), std::move(std::make_unique<ConstantNode<kernelT, evalT>>(B)));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator*(std::string A, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> B)
  {
    return std::make_unique<ProdNode<kernelT, evalT>>(std::move(std::make_unique<VariableNode<kernelT, evalT>>(A)), std::move(B));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator*(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A, std::string B)
  {
    return std::make_unique<ProdNode<kernelT, evalT>>(std::move(A), std::move(std::make_unique<VariableNode<kernelT, evalT>>(B)));
  };

  //operator/
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator/(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> B)
  {
    return std::make_unique<ProdNode<kernelT, evalT>>(std::move(A), std::make_unique<InvNode<kernelT, evalT>>(std::move(B)));
  };
  
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator/(std::string A, std::string B)
  {
    return std::make_unique<ProdNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(A), std::make_unique<InvNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(B)));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator/(evalT A, std::string B)
  {
    return std::make_unique<ProdNode<kernelT, evalT>>(std::make_unique<ConstantNode<kernelT, evalT>>(A), std::make_unique<InvNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(B)));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator/(std::string A, evalT B)
  {
    return std::make_unique<ProdNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(A), std::make_unique<InvNode<kernelT, evalT>>(std::make_unique<ConstantNode<kernelT, evalT>>(B)));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator/(evalT A, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> B)
  {
    return std::make_unique<ProdNode<kernelT, evalT>>(std::make_unique<ConstantNode<kernelT, evalT>>(A), std::make_unique<InvNode<kernelT, evalT>>(std::move(B)));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator/(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A, evalT B)
  {
    return std::make_unique<ProdNode<kernelT, evalT>>(std::move(A), std::make_unique<InvNode<kernelT, evalT>>(std::make_unique<ConstantNode<kernelT, evalT>>(B)));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator/(std::string A, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> B)
  {
    return std::make_unique<ProdNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(A), std::make_unique<InvNode<kernelT, evalT>>(std::move(B)));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator/(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A, std::string B)
  {
    return std::make_unique<ProdNode<kernelT, evalT>>(std::move(A), std::make_unique<InvNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(B)));
  };

  //operator-
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator-(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> B)
  {
    return std::make_unique<SumNode<kernelT, evalT>>(std::move(A), std::make_unique<NegNode<kernelT, evalT>>(std::move(B)));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator-(std::string A, std::string B)
  {
    return std::make_unique<SumNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(A), std::make_unique<NegNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(B)));
  };
  
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator-(std::string A, evalT B)
  {
    return std::make_unique<SumNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(A), std::make_unique<NegNode<kernelT, evalT>>(std::make_unique<ConstantNode<kernelT, evalT>>(B)));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator-(evalT A, std::string B)
  {
    return std::make_unique<SumNode<kernelT, evalT>>(std::make_unique<ConstantNode<kernelT, evalT>>(A), std::make_unique<NegNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(B)));
  };
    
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator-(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A, std::string B)
  {
    return std::make_unique<SumNode<kernelT, evalT>>(std::move(A), std::make_unique<NegNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(B)));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator-(std::string A, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> B)
  {
    return std::make_unique<SumNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(A), std::make_unique<NegNode<kernelT, evalT>>(std::move(B)));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator-(evalT A, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> B)
  {
    return std::make_unique<SumNode<kernelT, evalT>>(std::make_unique<ConstantNode<kernelT, evalT>>(A), std::make_unique<NegNode<kernelT, evalT>>(std::move(B)));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator-(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A, evalT B)
  {
    return std::make_unique<SumNode<kernelT, evalT>>(std::move(A), std::make_unique<NegNode<kernelT, evalT>>(std::make_unique<ConstantNode<kernelT, evalT>>(B)));
  };

  //operator- (single)
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator-(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A)
  {
    return std::make_unique<NegNode<kernelT, evalT>>(std::move(A));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator-(std::string A)
  {
    return std::make_unique<NegNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(A));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> operator-(evalT A)
  {
    return std::make_unique<NegNode<kernelT, evalT>>(std::make_unique<ConstantNode<kernelT, evalT>>(A));
  };

  //Inv
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Inv(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A)
  {
    return std::make_unique<InvNode<kernelT, evalT>>(std::move(A));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Inv(std::string A)
  {
    return std::make_unique<InvNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(A));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Inv(evalT A)
  {
    return std::make_unique<InvNode<kernelT, evalT>>(std::make_unique<ConstantNode<kernelT, evalT>>(A));
  };

  //Neg
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Neg(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A)
  {
    return std::make_unique<NegNode<kernelT, evalT>>(std::move(A));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Neg(std::string A)
  {
    return std::make_unique<NegNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(A));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Neg(evalT A)
  {
    return std::make_unique<NegNode<kernelT, evalT>>(std::make_unique<ConstantNode<kernelT, evalT>>(A));
  };

  //Sqrt
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Sqrt(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A)
  {
    return std::make_unique<SqrtNode<kernelT, evalT>>(std::move(A));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Sqrt(std::string A)
  {
    return std::make_unique<SqrtNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(A));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Sqrt(evalT A)
  {
    return std::make_unique<SqrtNode<kernelT, evalT>>(std::make_unique<ConstantNode<kernelT, evalT>>(A));
  };

  //Pow
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Pow(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A, std::unique_ptr<ComputeGraphNode<kernelT, evalT>> B)
  {
    return std::make_unique<PowNode<kernelT, evalT>>(std::move(A), std::move(B));
  };
  
  //Exp
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Exp(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A)
  {
    return std::make_unique<ExpNode<kernelT, evalT>>(std::move(A));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Exp(std::string A)
  {
    return std::make_unique<ExpNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(A));
  };
  
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Exp(evalT A)
  {
    return std::make_unique<ExpNode<kernelT, evalT>>(std::make_unique<ConstantNode<kernelT, evalT>>(A));
  };

  //Log
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Log(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A)
  {
    return std::make_unique<LogNode<kernelT, evalT>>(std::move(A));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Log(std::string A)
  {
    return std::make_unique<LogNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(A));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Log(evalT A)
  {
    return std::make_unique<LogNode<kernelT, evalT>>(std::move(std::make_unique<ConstantNode<kernelT, evalT>>(A)));
  };

  //Sin
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Sin(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A)
  {
    return std::make_unique<SinNode<kernelT, evalT>>(std::move(A));
  };
  
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Sin(std::string A)
  {
    return std::make_unique<SinNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(A));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Sin(evalT A)
  {
    return std::make_unique<SinNode<kernelT, evalT>>(std::make_unique<ConstantNode<kernelT, evalT>>(A));
  };

  //Cos
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Cos(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A)
  {
    return std::make_unique<CosNode<kernelT, evalT>>(std::move(A));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Cos(std::string A)
  {
    return std::make_unique<CosNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(A));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Cos(evalT A)
  {
    return std::make_unique<CosNode<kernelT, evalT>>(std::make_unique<ConstantNode<kernelT, evalT>>(A));
  };

  //Asin
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Asin(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A)
  {
    return std::make_unique<AsinNode<kernelT, evalT>>(std::move(A));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Asin(std::string A)
  {
    return std::make_unique<AsinNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(A));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Asin(evalT A)
  {
    return std::make_unique<AsinNode<kernelT, evalT>>(std::make_unique<ConstantNode<kernelT, evalT>>(A));
  };

  //Acos
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Acos(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A)
  {
    return std::make_unique<AcosNode<kernelT, evalT>>(std::move(A));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Acos(std::string A)
  {
    return std::make_unique<AcosNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(A));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Acos(evalT A)
  {
    return std::make_unique<AcosNode<kernelT, evalT>>(std::make_unique<ConstantNode<kernelT, evalT>>(A));
  };

  //Tan
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Tan(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A)
  {
    return std::make_unique<TanNode<kernelT, evalT>>(std::move(A));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Tan(std::string A)
  {
    return std::make_unique<TanNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(A));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Tan(evalT A)
  {
    return std::make_unique<TanNode<kernelT, evalT>>(std::make_unique<ConstantNode<kernelT, evalT>>(A));
  };

  //Atan
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Atan(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A)
  {
    return std::make_unique<AtanNode<kernelT, evalT>>(std::move(A));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Atan(std::string A)
  {
    return std::make_unique<AtanNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(A));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Atan(evalT A)
  {
    return std::make_unique<AtanNode<kernelT, evalT>>(std::make_unique<ConstantNode<kernelT, evalT>>(A));
  };

  //Erf
  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Erf(std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A)
  {
    return std::make_unique<ErfNode<kernelT, evalT>>(std::move(A));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Erf(std::string A)
  {
    return std::make_unique<ErfNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(A));
  };

  template<typename kernelT, typename evalT>
  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> Erf(evalT A)
  {
    return std::make_unique<ErfNode<kernelT, evalT>>(std::make_unique<ConstantNode<kernelT, evalT>>(A));
  };


}

#endif
