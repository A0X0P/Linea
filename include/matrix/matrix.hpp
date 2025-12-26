//created by : A.N. Prosper
//date : December 18
// th 2025
//time : 20:20

#ifndef MATRIX_H
#define MATRIX_H


#include <cassert>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>





template<typename M>
//requires std::is_integral_v<M> || std::is_floating_point_v<M>
class Matrix{

    private:
    //Attributes
    std::size_t row;
    std::size_t column;
    std::vector<M> data;
    

    public:
    //Constructors:

    //Matrix() = default;

   // shape-only initialization
    Matrix(std::size_t row, std::size_t column)
        : row(row), column(column), data(row * column, M{}){
    }

    //value-filled 
    Matrix(std::size_t row, std::size_t column, M value)
        : row(row), column(column), data(row * column, value){
    }

    //data-only initialization
    Matrix<M>(std::initializer_list<std::initializer_list<M>> list){
        this->row = list.size();
        this->column = list.begin()->size();
        //Matrix<int> mm = {};
        for (const auto& values : list) {
            assert(values.size() == this->column);
            this->data.insert(this->data.end(), values.begin(), values.end());
        }
    
    }


    //Destructor:

    ~Matrix(){
        //delete  T_matrix[column][row];
    }


    //Getters:

    std::size_t getRow()const{
        return row;
    }

    std::size_t getColumn() const{
        return column;
    }

    std::vector<M> getdata(){
        return data;
    }

    
    //Static Methods:

    //Indentity
    static Matrix<M> Indentity(std::size_t row_column) {
        Matrix<M> I(row_column, row_column);

        for (std::size_t i = 0; i < row_column; ++i) {
            I.data[i * row_column + i] = M{1};            
        }

        return I;
    }

   

    
    public:
    //Methods


    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>//


    //Operations:

    //equality
    bool operator==(Matrix<M> other) const noexcept{
        if ((this->row != other.row) || 
            (this->column != other.column)) {
                return false;
        }
        for (std::size_t i {}; i < this->data.size(); i++) {
            if (this->data[i] != other.data[i]) {
                return false;
            }
        }
        return true;
    };

    //addition
    Matrix<M> operator+(Matrix<M> other){
        
        if ((this->row != other.row) || (this->column != other.column)) {
            throw std::invalid_argument("Matrix addition requires identical dimensions.");
        }
        Matrix<M> result(this->row, this->column);

        for (std::size_t i = 0; i < this->row; ++i) {
            for (std::size_t j = 0; j < this->column; ++j) {
                result.data[i * this->column + j] =
                    this->data[i * this->column + j] + other.data[i * this->column + j];
            }
        }
        return result;
    }

    //subtraction
    Matrix<M> operator-(Matrix<M> other){
        
        if ((this->row != other.row) || 
            (this->column != other.column)) {
                throw std::invalid_argument("Matrix subtraction requires identical dimensions.");
        }
        Matrix<M> result(this->row, this->column);

        for (std::size_t i = 0; i < this->row; ++i) {
            for (std::size_t j = 0; j < this->column; ++j) {
                result.data[i * this->column + j] =
                    this->data[i * this->column + j] - other.data[i * this->column + j];
            }
        }
        return result;
    }

    //unary minus
    Matrix<M> operator-(){
        Matrix<M> result(row, column);
        for (auto& data_index : data) {
            for (auto& value : result.data) {
                value = (-M{1} * data_index);
            }
        }
        return result;
    }
    
    //scalar multiplication
    Matrix<M> operator*(M scalar) const{

        Matrix<M> result(this->row, this->column);
        for (auto& data_value : this->data) {
            for (auto& result_value : result.data) {
                result_value = (data_value * scalar);
            }
        }
        return result;
    }

    //scalar multiplication
    void scalar_Multiply(M scalar) const{
        for (auto& value : data) {
            value *= scalar;
        }
    }

    //matrix multiplication
    Matrix<M> operator*(const Matrix<M>& other) const{
        
        if (this->column != other.row) {
            throw std::invalid_argument("Matrix multiplication requires A.column() == B.row()");
        }
        Matrix<M> result(this->row, other.column);
        for (std::size_t i {}; i < this->row; i++) {
            for (std::size_t j {}; j < other.column; j++) {
                for (std::size_t k {}; k < other.row; k++) {
                    result.data[i * this->row + j] =
                    this->data[i * this->row + k] * other.data[j * this->row + k];
                }
            }
        }
        return result;
    }

    //index
    M& operator()(std::size_t i, std::size_t j) {
        if (i >= row || j >= column){
           throw std::out_of_range("Matrix index is out of range."); 
        }
        return  data[i * column + j];
    }


    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>//
    
    //Mathematical functions

    //logarithm
    Matrix<M> log(){

        Matrix<M> result(this->row, this->column);
        for (auto& value : this->data) {
            for (auto& log_value : result.data) {
                log_value = std::log(value);
            }
        }
        return result;
    }

    //power
    Matrix<M> pow(M k){
        
        Matrix<M> result(this->row, this->column);
        for (auto& value : this->data) {
            for (auto& pow_value : result.data) {
                pow_value = std::pow(value, k);
            }
        }
        return result;
    }

    //exponent
    Matrix<M> exp(){
        
        Matrix<M> result(this->row, this->column);
        for (auto& value : this->data) {
            for (auto& exp_value : result.data) {
                exp_value = std::exp(value);
            }
        }
        return result;
    }


    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>//

    //Special utility methods:

    //display
    void display(){
        std::cout << "displaying.." << std::endl;
        for (std::size_t i {}; i < row; i++){
            for (std::size_t j {}; j < column; j++){
                std::cout << data[i * column + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    //insert
    void insert(std::size_t row_index, std::size_t col_index, M number){
        if ((row_index >= 0 && row_index < row) && (col_index >= 0 && col_index < column)) {
            data[row_index * column + col_index] = number;
        }
        else {
            if (row_index < row){
                throw std::invalid_argument("Invalid column index");
            }
            else if (col_index < column){
                throw std::invalid_argument("Invalid row index");    
            }
            
        }
        
    }

    //shape
    void shape(){
        std::cout << "(" << row << ", " << column << ")" << std::endl;
    }

    //size
    int size(){
        return static_cast<int>(this->row * this->column);
    }

    //symmetric
    bool symmetric() const noexcept{
        if ((this->row != this->Transpose().row) || 
            (this->column != this->Transpose().column)){
                return false;
            }

        for (std::size_t i {}; i < this->data.size(); i++) {
                if (this->data[i] != this->Transpose().data[i]) {
                    return false;
                }
            }
        return true;
    }
    
    //diagonal
    M diagonal(){

        M major_diagonal = M{1};

        if(row != column){
            throw std::invalid_argument("Matrix diagonal requires row == column.");
        }
        for (std::size_t i {}; i < this->row; i++) { 
            for (std::size_t j {}; j < this->column; j++) {
                if (i == j) {
                    major_diagonal *= this->data[i * this->row + j];
                }
            }
        } 
        return major_diagonal;

    }

    //transpose
    Matrix<M> Transpose() const {
        Matrix<M> result(this->column, this->row);

        for (std::size_t i = 0; i < this->row; ++i) {
            for (std::size_t j = 0; j < this->column; ++j) {
                result.data[j * this->row + i] =
                    data[i * this->column + j];
            }
        }

        return result;
    }

    //Rank
    void Rank(){

    }

    //cofactor
    M cofactor(std::size_t i, std::size_t j){
        //((-1)^(i+j))*det(M(i,j))
        Matrix<M> minor(i, j);
        return std::pow(-1, (i+j)) * minor.determinant();

    }
    
    //adjoint
    Matrix<M> adjoint(){
        Matrix<M> result(this->row, this->column);

        for (std::size_t i {}; i < this->row; i++) {
            for (std::size_t j {}; j < this->column; j++) {
                result.data[i * this->row + j] = this->cofactor(i, j);
            }
        }
        return result.Transpose();
    }
    
    //determinant 
    void determinant(){

    }
    void Inverse(){

    }

    //LU DEcomposition with partial pivoting
    void LU_decompositon(){

    }





    

};


//scalar multiplication
template <typename T, typename Tp>
//requires std::is_integral_v<T> || std::is_floating_point_v<T>
Matrix<Tp> operator*(T scalar, Matrix<Tp>& matrix){
        
        Matrix<Tp> result(matrix.getRow(), matrix.getColumn());

        for (std::size_t i {}; i < result.getRow(); i++) {
            for (std::size_t j {}; j < result.getColumn(); j++) {
                result(i,j) = scalar * matrix(i,j);
            }
        }
        return result;
}



#endif



