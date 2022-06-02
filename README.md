numc

Provide answers to the following questions.
- How many hours did you spend on the following tasks?
- I spent a good amount of time, like more than 30 hours for rewatch lectures, reading intel guide, speeding it up and waiting for the autograder tokens :)
  - Task 1 (Matrix functions in C): 
  - 
        1.1: Get and Set 
          Returns the double value of the matrix at the given row and column.
          Sets the value at the given row and column to val.

        1.2: Allocate

          1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
          2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
          3. Allocate space for the matrix data, initializing all entries to be 0. Return -2 if allocating memory failed.
          4. Set the number of rows and columns in the matrix struct according to the arguments provided.
          5. Set the `parent` field to NULL, since this matrix was not created from a slice.
          6. Set the `ref_cnt` field to 1.
          7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
          8. Return 0 upon success.



        1.3: Deallocate
          1. If the matrix pointer `mat` is NULL, return.
          2. If `mat` has no parent: decrement its `ref_cnt` field by 1. If the `ref_cnt` field becomes 0, then free `mat` and its `data` field.
          3. Otherwise, recursively call `deallocate_matrix` on `mat`'s parent, then free `mat`
           It's pretty easy to get memory leak in this task. So I have to use CGDB to debug it. It was dreadful.

          1.4: Allocate Reference
              1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
             Allocate space for the new matrix struct. Return -2 if allocating memory failed.
             Set the `data` field of the new struct to be the `data` field of the `from` struct plus `offset`.
             Set the number of rows and columns in the new struct according to the arguments provided.
             Set the `parent` field of the new struct to the `from` struct pointer.
             Increment the `ref_cnt` field of the `from` struct by 1.
             Store the address of the allocated matrix struct at the location `mat` is pointing at.
             Return 0 upon success.

            Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
            It's also pretty easy to get memory leak in this task. So I have to use CGDB to debug it. It was dreadful.


          1.5: Basic Matrix Operations
          Fill in the fill_matrix, abs_matrix, and add_matrix functions in src/matrix.c.

          1.6: More Matrix Operations
          Fill in the mul_matrix function in src/matrix.c.

          Fill in the pow_matrix function in src/matrix.c. This function should multiply the mat matrix by itself repeatedly pow times. We recommend calling mul_matrix within this function.




  - Task 2 (Speeding up matrix operations): 
- Was this project interesting? What was the most interesting aspect about it?
  - It's a little bit work if you are not good at C and simd/openmd like me.
  - I had to rewatch all the lectures and read most of the intrinsic guide for this short project T_T (although it's already been cut off by a lot). 
- What did you learn?
  - I learned how to speedup the performance by using SIMD, Unrolling, OpenMD and other algorithm like repeated square 
- Is there anything you would change?
  - Using repeated square to speed up POW, maybe? I really want to try the recursive and repeated square for pow to speed it up, but I guess it's for after finals
Last but not least, thank you course staff :).
