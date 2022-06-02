#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid. Note that the matrix is in row-major order.
 */
double get(matrix *mat, int row, int col) {
    // Task 1.1 TODO
    while (mat != NULL ) {
        int column = mat->cols;
        double result = (mat->data)[row*(column) + col];
        return result;
    }
    return 0;
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid. Note that the matrix is in row-major order.
 */
void set(matrix *mat, int row, int col, double val) {
    // Task 1.1 TODO
    int column = mat->cols;
    mat->data[row * column + col] = val;
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    // Task 1.2 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
        if (rows < 1 || cols < 1 || mat == NULL) {
            return -1;
        }
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
        double *temp_data = calloc(rows*cols, sizeof(double));
        matrix *new_matrix = malloc(sizeof(matrix));
    // 3. Allocate space for the matrix data, initializing all entries to be 0. Return -2 if allocating memory failed.
        if(!new_matrix){
        return -2;
        }
    // 4. Set the number of rows and columns in the matrix struct according to the arguments provided.
        new_matrix->rows = rows;
        new_matrix->cols = cols;
        new_matrix->data = temp_data;
    // 5. Set the `parent` field to NULL, since this matrix was not created from a slice.
        new_matrix->parent = NULL;
    // 6. Set the `ref_cnt` field to 1.
        new_matrix->ref_cnt = 1;
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
        *mat = new_matrix;
    // 8. Return 0 upon success.
        return 0;

}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or that you free `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent
 * matrix has no other references (including itself).
 */
//Don't free too much 
void deallocate_matrix(matrix *mat) {
    // Task 1.3 TODO
    // HINTS: Follow these steps.
    // 1. If the matrix pointer `mat` is NULL, return.
    // 2. If `mat` has no parent: decrement its `ref_cnt` field by 1. If the `ref_cnt` field becomes 0, then free `mat` and its `data` field.
    // 3. Otherwise, recursively call `deallocate_matrix` on `mat`'s parent, then free `mat`.
        if (mat == NULL) {
        return;
        }
        if(mat->parent == NULL) {
            mat->ref_cnt -= 1;
            if(mat->ref_cnt == 0) {
                free(mat->data);
            }
        } else{
        deallocate_matrix(mat->parent);
        free(mat);
        }
}
        



/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`
 * and the reference counter for `from` should be incremented. Lastly, do not forget to set the
 * matrix's row and column values as well.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 * NOTE: Here we're allocating a matrix struct that refers to already allocated data, so
 * there is no need to allocate space for matrix data.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    // Task 1.4 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
        if (rows < 1 || cols < 1 || mat == NULL) {
                return -1;
            }
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
        matrix *new_matrix = malloc(sizeof(matrix));
        if(!new_matrix){
            return -2;
        }
    // 3. Set the `data` field of the new struct to be the `data` field of the `from` struct plus `offset`.
        new_matrix->data = from->data + offset;
    // 4. Set the number of rows and columns in the new struct according to the arguments provided.
        new_matrix->rows = rows;
        new_matrix->cols = cols;
    // 5. Set the `parent` field of the new struct to the `from` struct pointer.
        new_matrix->parent = from;
    // 6. Increment the `ref_cnt` field of the `from` struct by 1.
        from->ref_cnt += 1;
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
        *mat = new_matrix;
    // 8. Return 0 upon success.
        return 0;

}

/*
 * Sets all entries in mat to val. Note that the matrix is in row-major order.
 */
void fill_matrix(matrix *mat, double val) {
    // Task 1.5 TODO
    int ARRAY = (mat->rows)*(mat->cols);
    #pragma omp parallel for collapse(1)
    for (int i = 0; i < ARRAY; i++) {
        mat->data[i] = val;
    }
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int abs_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
    int ARRAY = (mat->rows)*(mat->cols);
    double *mat1_data = mat->data;
    double *result_data = result->data;
    __m256d load; 
    __m256d negative;
    __m256d value; 
    #pragma omp parallel for private(load, negative, value)
    for (int i = 0; i < ARRAY / 4 * 4; i += 4) {
        load = _mm256_loadu_pd(&(mat1_data[i]));
        negative = _mm256_sub_pd(_mm256_setzero_pd(), load);
        value = _mm256_and_pd(load, negative);
        _mm256_storeu_pd(&(result_data[i]), value);
    }
    //Tail Case
    for (int i = ARRAY / 4 * 4; i < ARRAY; i++) { // tail case
        double tail_data = mat1_data[i];
        if (tail_data > 0) {
            result_data[i] = tail_data;
        } else {
            result_data[i] = - tail_data;
        }
    }
    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int neg_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    int ARRAY_SIZE = (result->rows)*(result->cols);
    double *op_result = result->data;
    double *d_mat1 = mat1->data;
    double *d_mat2 = mat2->data;
    __m256d data1;
    __m256d data2;  
    __m256d sum;
    int stride = 16;
    // omp_set_num_threads(16);
    #pragma omp parallel for private(data1,data2, sum) collapse(1)
        for(int i = 0; i < ARRAY_SIZE/stride * stride; i+=stride) {
        data1 = _mm256_loadu_pd(d_mat1 + i);
        data2 = _mm256_loadu_pd(d_mat2 + i);
        sum = _mm256_add_pd(data1, data2);
        _mm256_storeu_pd(op_result + i, sum);

        data1 = _mm256_loadu_pd(d_mat1 + i + 4);
        data2 = _mm256_loadu_pd(d_mat2 + i + 4);
        sum = _mm256_add_pd(data1, data2);
        _mm256_storeu_pd(op_result + i+4, sum);

        data1 = _mm256_loadu_pd(d_mat1 + i + 8);
        data2 = _mm256_loadu_pd(d_mat2 + i + 8);
        sum = _mm256_add_pd(data1, data2);
        _mm256_storeu_pd(op_result + i+8, sum);

        data1 = _mm256_loadu_pd(d_mat1 + i + 12);
        data2 = _mm256_loadu_pd(d_mat2 + i + 12);        
        sum = _mm256_add_pd(data1, data2);
        _mm256_storeu_pd(op_result + i+12, sum);
    }
        for(int i = ARRAY_SIZE/ 16 * 16; i < ARRAY_SIZE; i+=1){
        op_result[i] = d_mat1[i] + d_mat2[i];
        }
    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 * You may assume `mat1`'s number of columns is equal to `mat2`'s number of rows.
 * Note that the matrix is in row-major order.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    double sum;
    matrix *mat2_T; //matrix_transpose
    double *op_result = result->data;
    double *d_mat1 = mat1->data;
    int r1 = mat1->rows;
    int r2 = mat2->rows;
    int c1 = mat1->cols;
    int c2 = mat2->cols;
    //STARTCODE
    allocate_matrix(&mat2_T, c2, c1);
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < c1; i++){
        for(int j = 0; j < c2; j++){
            (mat2_T->data)[j* c1 + i] = (mat2->data)[i * c2 + j];
	    }
    }
     //OPENMD
    __m256d data1;
    __m256d data2;  
    __m256d simd_sum;
    double *d_mat2 = mat2_T->data;
    // omp_set_num_threads(20);
    #pragma omp parallel for private(sum, data1, data2, simd_sum) collapse(2)
    for(int i = 0; i < r1; i++){
        for(int j = 0; j < c2; j++){
            simd_sum = _mm256_setzero_pd();
            __m256d product;
	        for(int k = 0; k < c1/20 * 20; k+=20){
                data1 = _mm256_loadu_pd(d_mat1+i*c1+k);
                data2 = _mm256_loadu_pd(d_mat2+j*c1+k);
                // product = _mm256_mul_pd(data1,data2);
                // simd_sum = _mm256_add_pd(product, simd_sum);
                simd_sum = _mm256_fmadd_pd(data1, data2, simd_sum);

                data1 = _mm256_loadu_pd(d_mat1+i*c1+k+4);
                data2 = _mm256_loadu_pd(d_mat2+j*c1+k+4);
                // product = _mm256_mul_pd(data1,data2);
                // simd_sum = _mm256_add_pd(product, simd_sum);
                simd_sum = _mm256_fmadd_pd(data1, data2, simd_sum);

                data1 = _mm256_loadu_pd(d_mat1+i*c1+k+8);
                data2 = _mm256_loadu_pd(d_mat2+j*c1+k+8);
                // product = _mm256_mul_pd(data1,data2);
                // simd_sum = _mm256_add_pd(product, simd_sum);
                simd_sum = _mm256_fmadd_pd(data1, data2, simd_sum);

                data1 = _mm256_loadu_pd(d_mat1+i*c1+k+12);
                data2 = _mm256_loadu_pd(d_mat2+j*c1+k+12);
                // product = _mm256_mul_pd(data1,data2);
                // simd_sum = _mm256_add_pd(product, simd_sum);
                simd_sum = _mm256_fmadd_pd(data1, data2, simd_sum);

                data1 = _mm256_loadu_pd(d_mat1+i*c1+k+16);
                data2 = _mm256_loadu_pd(d_mat2+j*c1+k+16);
                // product = _mm256_mul_pd(data1,data2);
                // simd_sum = _mm256_add_pd(product, simd_sum);
                simd_sum = _mm256_fmadd_pd(data1, data2, simd_sum);

                // data1 = _mm256_loadu_pd(d_mat1+i*c1+k+20);
                // data2 = _mm256_loadu_pd(d_mat2+j*c1+k+20);
                // // product = _mm256_mul_pd(data1,data2);
                // // simd_sum = _mm256_add_pd(product, simd_sum);
                // simd_sum = _mm256_fmadd_pd(data1, data2, simd_sum);

                // data1 = _mm256_loadu_pd(d_mat1+i*c1+k+24);
                // data2 = _mm256_loadu_pd(d_mat2+j*c1+k+24);
                // // product = _mm256_mul_pd(data1,data2);
                // // simd_sum = _mm256_add_pd(product, simd_sum);
                // simd_sum = _mm256_fmadd_pd(data1, data2, simd_sum);

                // data1 = _mm256_loadu_pd(d_mat1+i*c1+k+28);
                // data2 = _mm256_loadu_pd(d_mat2+j*c1+k+28);
                // product = _mm256_mul_pd(data1,data2);
                // simd_sum = _mm256_add_pd(product, simd_sum);
            }
            double temp[4];
            _mm256_storeu_pd(temp, simd_sum);
            //TAIL CASE
	        for(int k= c1/20 * 20; k < c1; k++){
                temp[0] += d_mat1[i * c1 + k] * d_mat2[j * c1 + k];
            }
            sum = temp[0] + temp[1] + temp[2] + temp[3];
	        op_result[i* c2 + j] += sum;
        }
    }
    deallocate_matrix(mat2_T);
    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 * You may assume `mat` is a square matrix and `pow` is a non-negative integer.
 * Note that the matrix is in row-major order.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    int ARRAY = (mat->rows)*(mat->cols);
    if (mat == NULL) {
        return -1;
    }
    if ((mat->rows) != (mat->cols)){
        return -1;
    }
    fill_matrix(result,0);
    #pragma omp parallel for
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            if (i == j) {
                result->data[mat->cols * i + j] = 1;
            } else {
                result->data[mat->cols * i + j] = 0;
            }
        } 
    }
    matrix *new_matrix;
    matrix *temp2;
    matrix *temp1;

    allocate_matrix(&new_matrix, mat->rows, mat->cols);
    #pragma omp parallel for
    for (int i = 0; i < ARRAY; i++) {
        new_matrix->data[i] = mat->data[i];
    }
    // #pragma omp parallel for
    // for (int i = 0; i < mat->rows; i++) {
    //             for (int j = 0; j < mat->cols; j++) {
    //                 new_matrix->data[i * mat->cols + j] = mat->data[i * mat->cols + j];
    //             }
    //         }
    while(pow) {
        if (pow % 2 == 1) { //Odd
            allocate_matrix(&temp1, mat->rows, mat->cols);
            mul_matrix(temp1, result, new_matrix);
            for (int i = 0; i < mat->rows; i++) {
                for (int j = 0; j < mat->cols; j++) {
                    result->data[i * mat->cols + j] = temp1->data[i * mat->cols + j];
                }
            }
        }
        allocate_matrix(&temp2, mat->rows, mat->cols);
        mul_matrix(temp2, new_matrix, new_matrix);
        for (int i = 0; i < mat->rows; i++) {
                for (int j = 0; j < mat->cols; j++) {
                    new_matrix->data[i * mat->cols + j] = temp2->data[i * mat->cols + j];
                }
            }
        pow >>= 1; //Even

    }
    deallocate_matrix(new_matrix);
    return 0;
}