package org.apache.sysml.runtime.io;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.SparseBlock;
import org.apache.sysml.runtime.matrix.data.SparseBlockMCSR;
import org.apache.sysml.runtime.matrix.mapred.MRJobConfiguration;
import org.apache.sysml.runtime.util.IndexRange;
import org.apache.sysml.udf.Matrix;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Created by Fei Hu on 8/4/17.
 */
public class ReaderSubBinaryBlackParallel extends ReaderBinaryBlock {

  private static int _numThreads = 1;

  public ReaderSubBinaryBlackParallel(boolean localFS) {
    super(localFS);
    _numThreads = OptimizerUtils.getParallelBinaryReadParallelism();
  }

  @Override
  public MatrixBlock readSubMatrixFromHDFS(String fname, long rlen, long clen, int brlen, int bclen,
                                        long estnnz, IndexRange ixRange)
      throws IOException, DMLRuntimeException {
    //allocate output matrix block (incl block allocation for parallel)
    long rlen_ret = Math.min((ixRange.rowEnd/brlen - ixRange.rowStart/brlen + 1) * brlen, rlen);
    long clen_ret = Math.min((ixRange.colEnd/bclen - ixRange.colStart/bclen + 1) * bclen, clen);
    MatrixBlock ret = createOutputMatrixBlock(rlen_ret, clen_ret, brlen, bclen, estnnz, true, true);

    //prepare file access
    JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
    Path path = new Path((_localFS ? "file:///" : "") + fname);
    FileSystem fs = IOUtilFunctions.getFileSystem(path, job);

    //check existence and non-empty file
    checkValidInputFile(fs, path);

    //core read
    readSubBinaryBlockMatrixFromHDFS(path, job, fs, ret, rlen, clen, brlen, bclen, ixRange);

    //finally check if change of sparse/dense block representation required
    if (!AGGREGATE_BLOCK_NNZ)
      ret.recomputeNonZeros();
    ret.examSparsity();

    IndexRange sub_ixRange = new IndexRange(ixRange.rowStart - ixRange.rowStart/brlen * brlen,
                                            ixRange.rowEnd - ixRange.rowStart/brlen * brlen,
                                            ixRange.colStart - ixRange.colStart/bclen * bclen,
                                            ixRange.colEnd - ixRange.colStart/bclen * bclen);

    return ret.sliceOperations(sub_ixRange, new MatrixBlock());
  }

  private static void readSubBinaryBlockMatrixFromHDFS(Path path, JobConf job, FileSystem fs,
                                                       MatrixBlock dest, long rlen, long clen,
                                                       int brlen, int bclen, IndexRange ixRange)
      throws IOException, DMLRuntimeException {
    //set up preferred custom serialization framework for binary block format
    if (MRJobConfiguration.USE_BINARYBLOCK_SERIALIZATION)
      MRJobConfiguration.addBinaryBlockSerializationFramework(job);

    try {
      //create read tasks for all files
      ExecutorService pool = Executors.newFixedThreadPool(_numThreads);
      ArrayList<ReaderSubBinaryBlackParallel.ReadFileTask>
          tasks = new ArrayList<ReaderSubBinaryBlackParallel.ReadFileTask>();
      for (Path lpath : IOUtilFunctions.getSequenceFilePaths(fs, path)) {
        ReaderSubBinaryBlackParallel.ReadFileTask
            t =
            new ReaderSubBinaryBlackParallel.ReadFileTask(lpath, job, fs, dest, rlen, clen, brlen,
                                                          bclen, ixRange);
        tasks.add(t);
      }

      //wait until all tasks have been executed
      List<Future<Object>> rt = pool.invokeAll(tasks);

      //check for exceptions and aggregate nnz
      long lnnz = 0;
      for (Future<Object> task : rt)
        lnnz += (Long) task.get();

      //post-processing
      dest.setNonZeros(lnnz);
      if (dest.isInSparseFormat() && clen > bclen)
        sortSparseRowsParallel(dest, rlen, _numThreads, pool);

      pool.shutdown();
    } catch (Exception e) {
      throw new IOException("Failed parallel read of binary block input.", e);
    }
  }

  private static class ReadFileTask implements Callable<Object> {

    private Path _path = null;
    private JobConf _job = null;
    private FileSystem _fs = null;
    private MatrixBlock _dest = null;
    private long _rlen = -1;
    private long _clen = -1;
    private int _brlen = -1;
    private int _bclen = -1;
    private IndexRange _ixRange = null;

    public ReadFileTask(Path path, JobConf job, FileSystem fs, MatrixBlock dest, long rlen,
                        long clen, int brlen, int bclen, IndexRange ixRange) {
      _path = path;
      _fs = fs;
      _job = job;
      _dest = dest;
      _rlen = rlen;
      _clen = clen;
      _brlen = brlen;
      _bclen = bclen;
      _ixRange = ixRange;
    }

    @Override
    @SuppressWarnings({"deprecation"})
    public Object call() throws Exception {
      boolean sparse = _dest.isInSparseFormat();
      MatrixIndexes key = new MatrixIndexes();
      MatrixBlock value = new MatrixBlock();
      long lnnz = 0; //aggregate block nnz

      //directly read from sequence files (individual partfiles)
      SequenceFile.Reader reader = new SequenceFile.Reader(_fs, _path, _job);

      try {
        System.out.println("****** SubMatrix Path is " + this._path.toString() + String
            .format(" row length %d; column length %d ", _rlen, _clen));
        //note: next(key, value) does not yet exploit the given serialization classes, record reader does but is generally slower.
        while (reader.next(key)) {
          int row_offset = (int) (key.getRowIndex() - 1) * _brlen;
          int col_offset = (int) (key.getColumnIndex() - 1) * _bclen;

          int min_row = (int) Math.max(row_offset, _ixRange.rowStart);
          int min_col = (int) Math.max(col_offset, _ixRange.colStart);
          int max_row = (int) Math.min(row_offset + _brlen, _ixRange.rowEnd);
          int max_col = (int) Math.min(col_offset + _bclen, _ixRange.colEnd);

          boolean isOverlapped = !((min_row > max_row) || (min_col > max_col));

          if (!isOverlapped) {
            System.out.println("******** Filter out the key " + key.toString());
            continue;
          }

            reader.getCurrentValue(value);
            int rows = value.getNumRows();
            int cols = value.getNumColumns();
            int row_offset_ret = (int) (row_offset - _ixRange.rowStart / _brlen * _brlen);
            int col_offset_ret = (int) (col_offset - _ixRange.colStart / _bclen * _bclen);

            System.out.println(String.format("------------ Key is (%d , %d) ~ (%d , %d), value size is %d", row_offset, col_offset, row_offset + rows, col_offset + cols, value.getInMemorySize()));

            //bound check per block
            if (row_offset + rows < 0 || row_offset + rows > _rlen || col_offset + cols < 0
                || col_offset + cols > _clen) {
              throw new IOException(
                  "Matrix block [" + (row_offset + 1) + ":" + (row_offset + rows) + "," + (col_offset
                                                                                           + 1) + ":"
                  + (col_offset + cols) + "] " +
                  "out of overall matrix range [1:" + _rlen + ",1:" + _clen + "].");
            }

            //copy block to result
            if (sparse) {
              //note: append requires final sort
              if (cols < _clen) {
                //sparse requires lock, when matrix is wider than one block
                //(fine-grained locking of block rows instead of the entire matrix)
                //NOTE: fine-grained locking depends on MCSR SparseRow objects
                SparseBlock sblock = _dest.getSparseBlock();
                if (sblock instanceof SparseBlockMCSR
                    && sblock.get(row_offset_ret) != null) {
                  synchronized (sblock.get(row_offset_ret)) {
                    _dest.appendToSparse(value, row_offset_ret, col_offset_ret);
                  }
                } else {
                  synchronized (_dest) {
                    _dest.appendToSparse(value, row_offset_ret, col_offset_ret);
                  }
                }
              } else { //quickpath (no synchronization)
                _dest.appendToSparse(value, row_offset_ret, col_offset_ret);
              }
            } else {
              _dest.copy(row_offset_ret, row_offset_ret + rows - 1,
                         col_offset_ret, col_offset_ret + cols - 1, value, false);
            }
            //aggregate nnz
            lnnz += value.getNonZeros();
          }
      } finally {
        IOUtilFunctions.closeSilently(reader);
      }

      return lnnz;
    }
  }
}