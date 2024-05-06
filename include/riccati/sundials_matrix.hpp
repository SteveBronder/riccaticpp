#ifndef INCLUDE_RICCATI_SUNDIALS_MATRIX_HPP
#define INCLUDE_RICCATI_SUNDIALS_MATRIX_HPP

#include <type_traits> 
#include <sundials/sundials_context.h>
#include <nvector/nvector_serial.h>
#include <sunlinsol/sunlinsol_dense.h>

namespace riccati {
namespace sundials_wrap {
template <long int Rows, long int Cols>
constexpr bool is_vector_v = Rows == 1 || Cols == 1;

template <typename T, long int CompileTimeRows_, long int CompileTimeCols_>
struct Vector {
    using index_t = long int;
    N_Vector nv_;

    Vector(sundials::Context ctx) : nv_(N_VNew_Serial(CompileTimeRows_ * CompileTimeCols_, ctx)) {}
    Vector(index_t size, sundials::Context ctx) : nv_(N_VNew_Serial(size, ctx)) {}
    Vector(N_Vector nv) : nv_(N_VClone(nv)) {}
    Vector(T* data, const index_t size, sundials::Context ctx) : nv_(N_VNewEmpty()) {
      N_VSetArrayPointer(data, nv_);
    }
    ~Vector() {
        N_VDestroy(nv_);
    }

    // Function to get the length of the vector
    long int length() const {
        return N_VGetLength(nv_);
    }

    // Function to get the raw data pointer
    T* data() {
        return static_cast<T*>(N_VGetArrayPointer(nv_));
    }

    // Overloaded operator to access elements of the vector
    T& operator()(long int i) {
        return NV_Ith_S(nv_, i);
    }

    // Overloaded operator to access elements of the vector
    const T& operator()(long int i) const {
        return NV_Ith_S(nv_, i);
    }
};

template <typename T, long int CompileTimeRows_, long int CompileTimeCols_>
struct Matrix {
    using index_t = long int;
    SUNMatrix sm_;

    Matrix(sundials::Context ctx) : sm_(SUNDenseMatrix(CompileTimeRows_ * CompileTimeCols_, ctx)) {}
    Matrix(index_t rows, index_t cols, sundials::Context ctx) : sm_(SUNDenseMatrix(rows * cols, ctx)) {}
    Matrix(T* data, const index_t rows, const index_t cols, sundials::Context ctx) : sm_(SUNDenseMatrix(rows * cols, ctx)) {
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          SM_ELEMENT_D(sm_, i, j) = data[j * rows + i];
        }
      }
    }
    ~Matrix() {
        SUNMatDestroy(sm_);
    }

    // Function to get the length of the Matrix
    const long int size() const {
        return SUNDenseMatrix_LData(sm_);
    }

    const long int rows() const {
        return SUNDenseMatrix_Rows(sm_);
    }

    const long int cols() const {
        return SUNDenseMatrix_Columns(sm_);
    }

    // Function to get the raw data pointer
    T* data() {
        return static_cast<T*>(SUNDenseMatrix_Data(sm_));
    }

    // Overloaded operator to access elements of the Matrix
    T& operator()(index_t i, index_t j) {
        return SM_ELEMENT_D(sm_, i, j);
    }

    // Overloaded operator to access elements of the Matrix
    const T& operator()(index_t i, index_t j) const {
        return SM_ELEMENT_D(sm_, i, j);
    }
};
}
}

#endif