import math


class CSR:

    def __init__(self):
        self.csr_dict = {'val': [], 'col_ind': [], 'context': [], 'row_ptr': [1]}
        self.val = self.csr_dict['val']
        self.col_ind = self.csr_dict['col_ind']
        self.row_ptr = self.csr_dict['row_ptr']
        self.context = self.csr_dict['context']
        self.rows = None
        self.columns = None
        self.nnz = None

    def build_model_numpy(self, np_array, user_list):
        col_lis = []
        self.rows = len(user_list)

        row = 0
        x, y = np_array.shape
        # Each user is represented once in user_list
        for index, user in enumerate(user_list):
            column = 0
            num_nz = 0
            while row < x and user == np_array[row][0]:
                if np_array[row][2] != 0:
                    self.csr_dict['col_ind'].append(column)
                    self.csr_dict['context'].append(np_array[row][2])
                    num_nz += 1
                column += 1
                row += 1
            col_lis.append(column)
            temp = num_nz + self.csr_dict['row_ptr'][index]
            self.csr_dict['row_ptr'].append(temp)

        self.nnz = self.csr_dict['row_ptr'][len(self.csr_dict['row_ptr']) - 1] - 1
        self.columns = max(col_lis)

    def build_no_context_numpy(self, np_array, user_list):
        col_lis = []
        self.rows = len(user_list)

        row = 0
        x, y = np_array.shape
        # Each user is represented once in user_list
        for index, user in enumerate(user_list):
            column = 0
            num_nz = 0
            while row < x and user == np_array[row][0]:

                dim_sum = 0
                # Magic numbers 4 and 12 represent the columns that our dimensions are found in
                # for(i = 4; i < 12; i++)
                for i in range(3, 11):
                    dim_sum += np_array[row][i]

                # If any context value is greater than 0, the sum clearly will not be equal to 0.
                if dim_sum == 0:
                    self.csr_dict['col_ind'].append(column)
                    self.csr_dict['val'].append(0)
                    self.csr_dict['context'].append(np_array[row][2])
                    num_nz += 1
                column += 1
                row += 1
            col_lis.append(column)
            temp = num_nz + self.csr_dict['row_ptr'][index]
            self.csr_dict['row_ptr'].append(temp)

        self.nnz = self.csr_dict['row_ptr'][len(self.csr_dict['row_ptr']) - 1] - 1
        self.columns = max(col_lis)

    def build_from_numpy(self, np_array, user_list, dimension):
        col_lis = []
        self.rows = len(user_list)

        row = 0
        x, y = np_array.shape
        # Each user is represented once in user_list
        for index, user in enumerate(user_list):
            column = 0
            num_nz = 0
            while row < x and user == np_array[row][0]:
                if np_array[row][dimension] != 0:
                    self.csr_dict['col_ind'].append(column)
                    self.csr_dict['val'].append(np_array[row][dimension])
                    self.csr_dict['context'].append(np_array[row][2])
                    num_nz += 1
                column += 1
                row += 1
            col_lis.append(column)
            temp = num_nz + self.csr_dict['row_ptr'][index]
            self.csr_dict['row_ptr'].append(temp)

        self.nnz = self.csr_dict['row_ptr'][len(self.csr_dict['row_ptr']) - 1] - 1
        self.columns = max(col_lis)

    def build_no_context(self, ws, user_list):
        col_lis = []
        self.rows = len(user_list)

        row = 2
        # Each user is represented once in user_list
        for index, user in enumerate(user_list):
            column = 0
            num_nz = 0
            while user == ws.cell(row=row, column=1).value:

                dim_sum = 0
                # Magic numbers 4 and 12 represent the columns that our dimensions are found in
                # for(i = 4; i < 12; i++)
                for i in range(4, 12):
                    dim_sum += ws.cell(row=row, column=i).value

                # If any context value is greater than 0, the sum clearly will not be equal to 0.
                if dim_sum == 0:
                    self.csr_dict['col_ind'].append(column)
                    self.csr_dict['val'].append(0)
                    self.csr_dict['context'].append(ws.cell(row=row, column=3).value)
                    num_nz += 1
                column += 1
                row += 1
            col_lis.append(column)
            temp = num_nz + self.csr_dict['row_ptr'][index]
            self.csr_dict['row_ptr'].append(temp)

        self.nnz = self.csr_dict['row_ptr'][len(self.csr_dict['row_ptr']) - 1] - 1
        self.columns = max(col_lis)

    def build_from_excel(self, ws, user_list, dimension):
        col_lis = []
        self.rows = len(user_list)

        row = 2
        # Each user is represented once in user_list
        for index, user in enumerate(user_list):
            column = 0
            num_nz = 0
            while user == ws.cell(row=row, column=1).value:
                if ws.cell(row=row, column=dimension).value != 0:
                    self.csr_dict['col_ind'].append(column)
                    self.csr_dict['val'].append(ws.cell(row=row, column=dimension).value)
                    self.csr_dict['context'].append(ws.cell(row=row, column=3).value)
                    num_nz += 1
                column += 1
                row += 1
            col_lis.append(column)
            temp = num_nz + self.csr_dict['row_ptr'][index]
            self.csr_dict['row_ptr'].append(temp)

        self.nnz = self.csr_dict['row_ptr'][len(self.csr_dict['row_ptr']) - 1] - 1
        self.columns = max(col_lis)

    def build_from_file(self, input_file):
        # Allow the user to build a new CSR without instantiating a new object
        self.csr_dict = {'val': [], 'col_ind': [], 'row_ptr': [1]}
        self.val = self.csr_dict['val']
        self.col_ind = self.csr_dict['col_ind']
        self.row_ptr = self.csr_dict['row_ptr']
        self.rows = None
        self.columns = None
        self.nnz = None

        # Create file object
        f = open(input_file, 'r')

        # Read the first line of the file and convert the values to integers
        # Use split with no argument. Using the argument " " caused the program to miss
        # 942 non zero values
        first_line = f.readline().strip('\n').split()
        self.rows = int(first_line[0])
        self.columns = int(first_line[1])
        self.nnz = int(first_line[2])

        # For all lines after the first line
        for line in f:
            # Assign line to array, strip the '\n' carriage and parse the space delimiter
            line_array = line.strip("\n").split()

            non_zero_counter = 0
            for (i, char) in enumerate(line_array):
                # There are instances of random spacing within the data, remove those items
                if line_array[i] == '':
                    del line_array[i]
                # If the index is even, append the (char - 1) to the col_ind array
                # We subtract 1 from the value to convert from MATLAB indexing to C indexing
                elif i % 2 == 0:
                    self.csr_dict['col_ind'].append(int(char) - 1)
                else:
                    self.csr_dict['val'].append(int(char))
                    non_zero_counter += 1

            row_ptr_len = len(self.csr_dict['row_ptr'])
            prev_row_ptr = self.csr_dict['row_ptr'][row_ptr_len - 1]
            self.csr_dict['row_ptr'].append(non_zero_counter + prev_row_ptr)

    def transpose(self,contextParam):
        nrows2 = self.rows
        ncols2 = self.columns
        nnz2 = self.nnz

        row_ptr2 = [0] * (ncols2 + 1)
        row_counts2 = [0] * ncols2

        col_ind2 = [None] * (nnz2 + 1)
        context2 = [None] * (nnz2 + 1)
        if (contextParam == True):
            val2 = [None] * (nnz2 + 1)

        # First run
        for i in range(0, self.rows):
            for j in range(self.row_ptr[i] - 1, self.row_ptr[i + 1] - 1):
                i2 = self.col_ind[j]
                row_ptr2[i2 + 1] += 1

        # A row_ptr always begins with 1 as the initial value
        row_ptr2[0] = 1
        # Create the final row_ptr list. Add the ith position to the ith - 1 position, with
        # 1 <= i <= length of the array
        for i in range(1, len(row_ptr2)):
            row_ptr2[i] += row_ptr2[i-1]


        # Second run
        for i in range(0, self.rows):
            for j in range(self.row_ptr[i] - 1, self.row_ptr[i + 1] - 1):
                i2 = self.col_ind[j]
                col_ind2[row_ptr2[i2] + row_counts2[i2]] = i
                context2[row_ptr2[i2] + row_counts2[i2]] = self.context[j]
                if (contextParam == True):
                    val2[row_ptr2[i2] + row_counts2[i2]] = self.val[j]
                row_counts2[i2] += 1

        del context2[0]
        del col_ind2[0]
        if (contextParam == True):
            del val2[0]

        # We will want to return a new CSR() and set the member variables appropriately
        transposed_csr = CSR()
        transposed_csr.context = context2
        transposed_csr.col_ind = col_ind2
        transposed_csr.row_ptr = row_ptr2
        if (contextParam == True):
            transposed_csr.val = val2
        transposed_csr.rows = self.columns
        transposed_csr.columns = self.rows
        transposed_csr.nnz = self.nnz
        transposed_csr.csr_dict = {'context': transposed_csr.context, 'val': transposed_csr.val, 'col_ind': transposed_csr.col_ind, 'row_ptr': transposed_csr.row_ptr}

        return transposed_csr

    def output_matrix(self, output_file):

        f = open(output_file, 'w')
        # Write the "row column #nonzero" line
        f.write(str(self.rows) + " " + str(self.columns) + " " + str(self.nnz) + "\n")

        # Determine the number of non zero values in a row
        # Append the values to a list
        number_of_vals_array = []
        for i in range(1, len(self.row_ptr)):
            number_of_vals = self.row_ptr[i] - self.row_ptr[i -1]
            number_of_vals_array.append(number_of_vals)

        vals_read = 0
        # For each row in the matrix
        for i, val in enumerate(number_of_vals_array):
            # For the number of non zero values in the row, write the respective col_ind and values
            for j in range(0, val):
                # We keep track of a vals_read so that we can properly increment along the lists
                # If we used j as an index, we duplicate the writing of elements
                f.write(str(self.col_ind[vals_read] + 1) + " " + str(self.val[vals_read]) + " ")
                vals_read += 1
            f.write("\n")

        #print("File written: " + output_file)

    def calculate_cosine_sim(self):

        sim_csr = CSR()

        # For each row in the matrix, compute the cosine sim with all pairs of
        # the matrix
        for i in range(0, len(self.row_ptr) - 1):
            nrowi = self.row_ptr[i + 1] - self.row_ptr[i]

            sim_nnz = 0
            for j in range(0, len(self.row_ptr) - 1):
                nrowj = self.row_ptr[j + 1] - self.row_ptr[j]

                # Reset values with each new pair
                ni = 0
                nj = 0
                cosine = 0
                lengthi = 0
                lengthj = 0

                while (ni < nrowi) and (nj < nrowj):

                    # Subtract by 1 to start from the 0 index when i, j = 0
                    ci = self.row_ptr[i] - 1 + ni
                    cj = self.row_ptr[j] - 1 + nj

                    if self.col_ind[ci] == self.col_ind[cj]:
                        # Added check
                        if self.context[ci] == self.context[cj]:
                            cosine += self.val[ci] * self.val[cj]
                        lengthi += self.val[ci] ** 2
                        lengthj += self.val[cj] ** 2

                        ni += 1
                        nj += 1

                    # If the col_ind[ci] gets ahead of col_ind[cj], increment nj
                    # to progress col_ind[cj]
                    elif self.col_ind[ci] > self.col_ind[cj]:
                        lengthj += self.val[cj] ** 2
                        nj += 1

                    # If the col_ind[cj] gets ahead of col_ind[ci], increment ni
                    # to progress col_ind[ci]
                    elif self.col_ind[ci] < self.col_ind[cj]:
                        lengthi += self.val[ci] ** 2
                        ni += 1

                # If nj == nrowj before ni == nrowi, finish the computations for ni
                # until ni == nrowi
                while ni < nrowi:
                    ci = self.row_ptr[i] - 1 + ni
                    lengthi += self.val[ci] ** 2
                    ni += 1

                # If ni == nrowi before nj == nrowj, finish the computations for nj
                # until nj == nrowj
                while nj < nrowj:
                    cj = self.row_ptr[j] - 1 + nj
                    lengthj += self.val[cj] ** 2
                    nj += 1

                # Calculate the similarity combining the values from the while loops
                if lengthi * lengthj:
                    cosine /= math.sqrt(lengthi * lengthj)
                else:
                    cosine = 0

                if cosine > 0:
                    sim_csr.csr_dict['val'].append(cosine)
                    # j + 1 preserves Matlab indexing
                    sim_csr.csr_dict['col_ind'].append(j + 1)
                    sim_nnz += 1

            sim_csr.csr_dict['row_ptr'].append(sim_csr.csr_dict['row_ptr'][i] + sim_nnz)

        return sim_csr

    def calculate_and_output_cosine_sim(self, sim_out, threshold):
        # We will be appending the file ultimately, this will clear the file
        f = open(sim_out, 'w')
        f.close()

        # For each row in the matrix, compute the cosine sim with all pairs of
        # the matrix
        for i in range(0, len(self.row_ptr) - 1):
            nrowi = self.row_ptr[i + 1] - self.row_ptr[i]

            for j in range(0, len(self.row_ptr) - 1):
                nrowj = self.row_ptr[j + 1] - self.row_ptr[j]

                # Reset values with each new pair
                ni = 0
                nj = 0
                cosine = 0
                lengthi = 0
                lengthj = 0

                while (ni < nrowi) and (nj < nrowj):

                    # Subtract by 1 to start from the 0 index when i, j = 0
                    ci = self.row_ptr[i] - 1 + ni
                    cj = self.row_ptr[j] - 1 + nj

                    if self.col_ind[ci] == self.col_ind[cj]:
                        # Added check
                        if self.context[ci] == self.context[cj] and self.context[ci] > 0:
                            cosine += self.val[ci] * self.val[cj]
                        lengthi += self.val[ci] ** 2
                        lengthj += self.val[cj] ** 2

                        ni += 1
                        nj += 1

                    # If the col_ind[ci] gets ahead of col_ind[cj], increment nj
                    # to progress col_ind[cj]
                    elif self.col_ind[ci] > self.col_ind[cj]:
                        lengthj += self.val[cj] ** 2
                        nj += 1

                    # If the col_ind[cj] gets ahead of col_ind[ci], increment ni
                    # to progress col_ind[ci]
                    elif self.col_ind[ci] < self.col_ind[cj]:
                        lengthi += self.val[ci] ** 2
                        ni += 1

                # If nj == nrowj before ni == nrowi, finish the computations for ni
                # until ni == nrowi
                while ni < nrowi:
                    ci = self.row_ptr[i] - 1 + ni
                    lengthi += self.val[ci] ** 2
                    ni += 1

                # If ni == nrowi before nj == nrowj, finish the computations for nj
                # until nj == nrowj
                while nj < nrowj:
                    cj = self.row_ptr[j] - 1 + nj
                    lengthj += self.val[cj] ** 2
                    nj += 1

                # Calculate the similarity combining the values from the while loops
                if lengthi * lengthj:
                    cosine /= math.sqrt(lengthi * lengthj)
                else:
                    cosine = 0

                # If the cosine is greater than the threshold parameter,
                # append the value to the text file
                if cosine > float(threshold):
                    f = open(sim_out, 'a')
                    # Increment the row/column values by 1 for MatLab style indexing
                    f.write(str(i + 1) + " " + str(j + 1) + " " + str(cosine) + "\n")

        print("File written: " + sim_out)

    def calc_user_user_sim(self, userI, userJ):
        nrowi = self.row_ptr[userI] - self.row_ptr[userI -1]
        nrowj = self.row_ptr[userJ] - self.row_ptr[userJ -1]
        ni = 0
        nj = 0
        cosine = 0
        lengthi = 0
        lengthj = 0

        while (ni < nrowi) and (nj < nrowj):

            # Subtract by 1 to start from the 0 index when i, j = 0
            ci = self.row_ptr[userI - 1] - 1 + ni
            cj = self.row_ptr[userJ - 1] - 1 + nj

            if self.col_ind[ci] == self.col_ind[cj]:
                # Added check
                if self.context[ci] == self.context[cj] and self.context[ci] > 0:
                    cosine += self.val[ci] * self.val[cj]
                lengthi += self.val[ci] ** 2
                lengthj += self.val[cj] ** 2

                ni += 1
                nj += 1

            # If the col_ind[ci] gets ahead of col_ind[cj], increment nj
            # to progress col_ind[cj]
            elif self.col_ind[ci] > self.col_ind[cj]:
                lengthj += self.val[cj] ** 2
                nj += 1

            # If the col_ind[cj] gets ahead of col_ind[ci], increment ni
            # to progress col_ind[ci]
            elif self.col_ind[ci] < self.col_ind[cj]:
                lengthi += self.val[ci] ** 2
                ni += 1

        # If nj == nrowj before ni == nrowi, finish the computations for ni
        # until ni == nrowi
        while ni < nrowi:
            ci = self.row_ptr[userI - 1] - 1 + ni
            lengthi += self.val[ci] ** 2
            ni += 1

        # If ni == nrowi before nj == nrowj, finish the computations for nj
        # until nj == nrowj
        while nj < nrowj:
            cj = self.row_ptr[userJ -1] - 1 + nj
            lengthj += self.val[cj] ** 2
            nj += 1

        # Calculate the similarity combining the values from the while loops
        if lengthi * lengthj:
            cosine /= math.sqrt(lengthi * lengthj)
        else:
            cosine = 0

        return cosine


