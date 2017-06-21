/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.runtime.controlprogram.parfor;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map.Entry;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.CacheException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.parfor.util.Cell;
import org.apache.sysml.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysml.runtime.controlprogram.parfor.util.StagingFileUtils;
import org.apache.sysml.runtime.io.IOUtilFunctions;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixFormatMetaData;
import org.apache.sysml.runtime.matrix.data.IJV;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixCell;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.runtime.util.FastStringTokenizer;
import org.apache.sysml.runtime.util.LocalFileUtils;
import org.apache.sysml.runtime.util.MapReduceTool;

/**
 * 
 * TODO potential extension: parallel merge (create individual staging files concurrently)
 *     
 *      NOTE: file merge typically used due to memory constraints - parallel merge would increase the memory
 *      consumption again.
 */
public class ResultMergeLocalFile extends ResultMerge
{
	
	//NOTE: if we allow simple copies, this might result in a scattered file and many MR tasks for subsequent jobs
	public static final boolean ALLOW_COPY_CELLFILES = false;	
	
	//internal comparison matrix
	private IDSequence _seq = null;
	
	public ResultMergeLocalFile( MatrixObject out, MatrixObject[] in, String outputFilename )
	{
		super( out, in, outputFilename );
		
		_seq = new IDSequence();
	}


	@Override
	public MatrixObject executeSerialMerge() 
		throws DMLRuntimeException 
	{
		MatrixObject moNew = null; //always create new matrix object (required for nested parallelism)

		//Timing time = null;
		LOG.trace("ResultMerge (local, file): Execute serial merge for output "+_output.getVarName()+" (fname="+_output.getFileName()+")");
		//	time = new Timing();
		//	time.start();

		
		try
		{
			
			
			//collect all relevant inputs
			ArrayList<MatrixObject> inMO = new ArrayList<MatrixObject>();
			for( MatrixObject in : _inputs )
			{
				//check for empty inputs (no iterations executed)
				if( in !=null && in != _output ) 
				{
					//ensure that input file resides on disk
					in.exportData();
					
					//add to merge list
					inMO.add( in );
				}
			}

			if( !inMO.isEmpty() )
			{
				//ensure that outputfile (for comparison) resides on disk
				_output.exportData();
				
				//actual merge
				merge( _outputFName, _output, inMO );
				
				//create new output matrix (e.g., to prevent potential export<->read file access conflict
				moNew = createNewMatrixObject( _output, inMO );	
			}
			else
			{
				moNew = _output; //return old matrix, to prevent copy
			}
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}

		//LOG.trace("ResultMerge (local, file): Executed serial merge for output "+_output.getVarName()+" (fname="+_output.getFileName()+") in "+time.stop()+"ms");
		
		return moNew;
	}
	
	@Override
	public MatrixObject executeParallelMerge(int par) 
		throws DMLRuntimeException 
	{
		//graceful degradation to serial merge
		return executeSerialMerge();
	}

	private MatrixObject createNewMatrixObject(MatrixObject output, ArrayList<MatrixObject> inMO ) 
		throws DMLRuntimeException
	{
		String varName = _output.getVarName();
		ValueType vt = _output.getValueType();
		MatrixFormatMetaData metadata = (MatrixFormatMetaData) _output.getMetaData();
		
		MatrixObject moNew = new MatrixObject( vt, _outputFName );
		moNew.setVarName( varName.contains(NAME_SUFFIX) ? varName : varName+NAME_SUFFIX );
		moNew.setDataType( DataType.MATRIX );
		
		//create deep copy of metadata obj
		MatrixCharacteristics mcOld = metadata.getMatrixCharacteristics();
		OutputInfo oiOld = metadata.getOutputInfo();
		InputInfo iiOld = metadata.getInputInfo();
		MatrixCharacteristics mc = new MatrixCharacteristics(mcOld.getRows(),mcOld.getCols(),
				                                             mcOld.getRowsPerBlock(),mcOld.getColsPerBlock());
		mc.setNonZeros( computeNonZeros(output, inMO) );
		MatrixFormatMetaData meta = new MatrixFormatMetaData(mc,oiOld,iiOld);
		moNew.setMetaData( meta );
		
		return moNew;
	}

	private void merge( String fnameNew, MatrixObject outMo, ArrayList<MatrixObject> inMO ) 
		throws DMLRuntimeException
	{
		OutputInfo oi = ((MatrixFormatMetaData)outMo.getMetaData()).getOutputInfo();
		boolean withCompare = ( outMo.getNnz() != 0 ); //if nnz exist or unknown (-1)
		
		if( oi == OutputInfo.TextCellOutputInfo )
		{
			if(withCompare)
				mergeTextCellWithComp(fnameNew, outMo, inMO);
			else
				mergeTextCellWithoutComp( fnameNew, outMo, inMO );
		}
		else if( oi == OutputInfo.BinaryCellOutputInfo )
		{
			if(withCompare)
				mergeBinaryCellWithComp(fnameNew, outMo, inMO);
			else
				mergeBinaryCellWithoutComp( fnameNew, outMo, inMO );
		}
		else if( oi == OutputInfo.BinaryBlockOutputInfo )
		{
			if(withCompare)
				mergeBinaryBlockWithComp( fnameNew, outMo, inMO );
			else
				mergeBinaryBlockWithoutComp( fnameNew, outMo, inMO );
		}
	}

	private void mergeTextCellWithoutComp( String fnameNew, MatrixObject outMo, ArrayList<MatrixObject> inMO ) 
		throws DMLRuntimeException
	{
		try
		{
			//delete target file if already exists
			MapReduceTool.deleteFileIfExistOnHDFS(fnameNew);
			
			if( ALLOW_COPY_CELLFILES )
			{
				copyAllFiles(fnameNew, inMO);
				return; //we're done
			}
			
			//actual merge
			JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
			Path path = new Path( fnameNew );
			FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
			BufferedWriter out = new BufferedWriter(new OutputStreamWriter(fs.create(path,true)));		
			
			String valueStr = null;
			
			try
			{
				for( MatrixObject in : inMO ) //read/write all inputs
				{
					LOG.trace("ResultMerge (local, file): Merge input "+in.getVarName()+" (fname="+in.getFileName()+") via stream merge");
					
					JobConf tmpJob = new JobConf(ConfigurationManager.getCachedJobConf());
					Path tmpPath = new Path(in.getFileName());
					FileInputFormat.addInputPath(tmpJob, tmpPath);
					TextInputFormat informat = new TextInputFormat();
					informat.configure(tmpJob);
					InputSplit[] splits = informat.getSplits(tmpJob, 1);
					
					LongWritable key = new LongWritable();
					Text value = new Text();
		
					for(InputSplit split: splits)
					{
						RecordReader<LongWritable,Text> reader = informat.getRecordReader(split, tmpJob, Reporter.NULL);
						try
						{
							while(reader.next(key, value))
							{
								valueStr = value.toString().trim();	
								out.write( valueStr+"\n" );
							}
						}
						finally {
							IOUtilFunctions.closeSilently(reader);
						}
					}
				}
			}
			finally {
				IOUtilFunctions.closeSilently(out);
			}
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException("Unable to merge text cell results.", ex);
		}
	}

	private void mergeTextCellWithComp( String fnameNew, MatrixObject outMo, ArrayList<MatrixObject> inMO ) 
		throws DMLRuntimeException
	{
		String fnameStaging = LocalFileUtils.getUniqueWorkingDir(LocalFileUtils.CATEGORY_RESULTMERGE);
		String fnameStagingCompare = LocalFileUtils.getUniqueWorkingDir(LocalFileUtils.CATEGORY_RESULTMERGE);
		
		try
		{
			//delete target file if already exists
			MapReduceTool.deleteFileIfExistOnHDFS(fnameNew);
			
			//Step 0) write compare blocks to staging area (if necessary)
			LOG.trace("ResultMerge (local, file): Create merge compare matrix for output "+outMo.getVarName()+" (fname="+outMo.getFileName()+")");
			createTextCellStagingFile(fnameStagingCompare, outMo, 0);
			
			//Step 1) read and write blocks to staging area
			for( MatrixObject in : inMO )
			{
				LOG.trace("ResultMerge (local, file): Merge input "+in.getVarName()+" (fname="+in.getFileName()+")");
				
				long ID = _seq.getNextID();
				createTextCellStagingFile( fnameStaging, in, ID );
			}
	
			//Step 2) read blocks, consolidate, and write to HDFS
			createTextCellResultFile(fnameStaging, fnameStagingCompare, fnameNew, (MatrixFormatMetaData)outMo.getMetaData(), true);
		}	
		catch(Exception ex)
		{
			throw new DMLRuntimeException("Unable to merge text cell results.", ex);
		}
		
		LocalFileUtils.cleanupWorkingDirectory(fnameStaging);
		LocalFileUtils.cleanupWorkingDirectory(fnameStagingCompare);
	}

	@SuppressWarnings("deprecation")
	private void mergeBinaryCellWithoutComp( String fnameNew, MatrixObject outMo, ArrayList<MatrixObject> inMO ) 
		throws DMLRuntimeException
	{
		try
		{	
			//delete target file if already exists
			MapReduceTool.deleteFileIfExistOnHDFS(fnameNew);
			
			if( ALLOW_COPY_CELLFILES )
			{
				copyAllFiles(fnameNew, inMO);
				return; //we're done
			}
			
			//actual merge
			JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
			Path path = new Path( fnameNew );					
			FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
			SequenceFile.Writer out = new SequenceFile.Writer(fs, job, path, MatrixIndexes.class, MatrixCell.class); //beware ca 50ms
			
			MatrixIndexes key = new MatrixIndexes();
			MatrixCell value = new MatrixCell();
			
			try
			{
				for( MatrixObject in : inMO ) //read/write all inputs
				{
					LOG.trace("ResultMerge (local, file): Merge input "+in.getVarName()+" (fname="+in.getFileName()+") via stream merge");
					
					JobConf tmpJob = new JobConf(ConfigurationManager.getCachedJobConf());
					Path tmpPath = new Path(in.getFileName());
					
					for(Path lpath : IOUtilFunctions.getSequenceFilePaths(fs, tmpPath) )
					{
						SequenceFile.Reader reader = new SequenceFile.Reader(fs,lpath,tmpJob);
						try
						{
							while(reader.next(key, value))
							{
								out.append(key, value);
							}
						}
						finally {
							IOUtilFunctions.closeSilently(reader);
						}
					}					
				}	
			}
			finally {
				IOUtilFunctions.closeSilently(out);
			}
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException("Unable to merge binary cell results.", ex);
		}	
	}

	private void mergeBinaryCellWithComp( String fnameNew, MatrixObject outMo, ArrayList<MatrixObject> inMO ) 
		throws DMLRuntimeException
	{
		String fnameStaging = LocalFileUtils.getUniqueWorkingDir(LocalFileUtils.CATEGORY_RESULTMERGE);
		String fnameStagingCompare = LocalFileUtils.getUniqueWorkingDir(LocalFileUtils.CATEGORY_RESULTMERGE);
		
		try
		{
			//delete target file if already exists
			MapReduceTool.deleteFileIfExistOnHDFS(fnameNew);
			
			//Step 0) write compare blocks to staging area (if necessary)
			LOG.trace("ResultMerge (local, file): Create merge compare matrix for output "+outMo.getVarName()+" (fname="+outMo.getFileName()+")");
			createBinaryCellStagingFile(fnameStagingCompare, outMo, 0);
			
			//Step 1) read and write blocks to staging area
			for( MatrixObject in : inMO )
			{
				LOG.trace("ResultMerge (local, file): Merge input "+in.getVarName()+" (fname="+in.getFileName()+")");
				
				long ID = _seq.getNextID();
				createBinaryCellStagingFile( fnameStaging, in, ID );
			}
	
			//Step 2) read blocks, consolidate, and write to HDFS
			createBinaryCellResultFile(fnameStaging, fnameStagingCompare, fnameNew, (MatrixFormatMetaData)outMo.getMetaData(), true);
		}	
		catch(Exception ex)
		{
			throw new DMLRuntimeException("Unable to merge binary cell results.", ex);
		}
		
		LocalFileUtils.cleanupWorkingDirectory(fnameStaging);
		LocalFileUtils.cleanupWorkingDirectory(fnameStagingCompare);
	}

	private void mergeBinaryBlockWithoutComp( String fnameNew, MatrixObject outMo, ArrayList<MatrixObject> inMO ) 
		throws DMLRuntimeException
	{
		String fnameStaging = LocalFileUtils.getUniqueWorkingDir(LocalFileUtils.CATEGORY_RESULTMERGE);
		
		try
		{
			//delete target file if already exists
			MapReduceTool.deleteFileIfExistOnHDFS(fnameNew);
			
			//Step 1) read and write blocks to staging area
			for( MatrixObject in : inMO )
			{
				LOG.trace("ResultMerge (local, file): Merge input "+in.getVarName()+" (fname="+in.getFileName()+")");				
				
				createBinaryBlockStagingFile( fnameStaging, in );
			}
	
			//Step 2) read blocks, consolidate, and write to HDFS
			createBinaryBlockResultFile(fnameStaging, null, fnameNew, (MatrixFormatMetaData)outMo.getMetaData(), false);
		}	
		catch(Exception ex)
		{
			throw new DMLRuntimeException("Unable to merge binary block results.", ex);
		}	
		
		LocalFileUtils.cleanupWorkingDirectory(fnameStaging);
	}

	private void mergeBinaryBlockWithComp( String fnameNew, MatrixObject outMo, ArrayList<MatrixObject> inMO ) 
		throws DMLRuntimeException
	{
		String fnameStaging = LocalFileUtils.getUniqueWorkingDir(LocalFileUtils.CATEGORY_RESULTMERGE);
		String fnameStagingCompare = LocalFileUtils.getUniqueWorkingDir(LocalFileUtils.CATEGORY_RESULTMERGE);
		
		try
		{
			//delete target file if already exists
			MapReduceTool.deleteFileIfExistOnHDFS(fnameNew);
			
			//Step 0) write compare blocks to staging area (if necessary)
			LOG.trace("ResultMerge (local, file): Create merge compare matrix for output "+outMo.getVarName()+" (fname="+outMo.getFileName()+")");			
			
			createBinaryBlockStagingFile(fnameStagingCompare, outMo);
			
			//Step 1) read and write blocks to staging area
			for( MatrixObject in : inMO )
			{
				LOG.trace("ResultMerge (local, file): Merge input "+in.getVarName()+" (fname="+in.getFileName()+")");		
				createBinaryBlockStagingFile( fnameStaging, in );
			}
	
			//Step 2) read blocks, consolidate, and write to HDFS
			createBinaryBlockResultFile(fnameStaging, fnameStagingCompare, fnameNew, (MatrixFormatMetaData)outMo.getMetaData(), true);
		}	
		catch(Exception ex)
		{
			throw new DMLRuntimeException("Unable to merge binary block results.", ex);
		}	
		
		LocalFileUtils.cleanupWorkingDirectory(fnameStaging);
		LocalFileUtils.cleanupWorkingDirectory(fnameStagingCompare);
	}

	@SuppressWarnings("deprecation")
	private void createBinaryBlockStagingFile( String fnameStaging, MatrixObject mo ) 
		throws IOException
	{		
		MatrixIndexes key = new MatrixIndexes(); 
		MatrixBlock value = new MatrixBlock();
		
		JobConf tmpJob = new JobConf(ConfigurationManager.getCachedJobConf());
		Path tmpPath = new Path(mo.getFileName());
		FileSystem fs = IOUtilFunctions.getFileSystem(tmpPath, tmpJob);
		
		for(Path lpath : IOUtilFunctions.getSequenceFilePaths(fs, tmpPath))
		{
			SequenceFile.Reader reader = new SequenceFile.Reader(fs,lpath,tmpJob);
			try
			{
				while(reader.next(key, value)) //for each block
				{							
					String lname = key.getRowIndex()+"_"+key.getColumnIndex();
					String dir = fnameStaging+"/"+lname;
					if( value.getNonZeros()>0 ) //write only non-empty blocks
					{
						LocalFileUtils.checkAndCreateStagingDir( dir );
						LocalFileUtils.writeMatrixBlockToLocal(dir+"/"+_seq.getNextID(), value);
					}
				}
			}
			finally {
				IOUtilFunctions.closeSilently(reader);
			}
		}
	}

	private void createTextCellStagingFile( String fnameStaging, MatrixObject mo, long ID ) 
		throws IOException, DMLRuntimeException
	{		
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(mo.getFileName());
		FileInputFormat.addInputPath(job, path);
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		InputSplit[] splits = informat.getSplits(job, 1);
		
		LinkedList<Cell> buffer = new LinkedList<Cell>();
		LongWritable key = new LongWritable();
		Text value = new Text();

		MatrixCharacteristics mc = mo.getMatrixCharacteristics();
		int brlen = mc.getRowsPerBlock(); 
		int bclen = mc.getColsPerBlock();
		//long row = -1, col = -1; //FIXME needs reconsideration whenever textcell is used actively
		//NOTE MB: Originally, we used long row, col but this led reproducibly to JIT compilation
		// errors during runtime; experienced under WINDOWS, Intel x86-64, IBM JDK 64bit/32bit.
		// It works fine with int row, col but we require long for larger matrices.
		// Since, textcell is never used for result merge (hybrid/hadoop: binaryblock, singlenode:binarycell)
		// we just propose the to exclude it with -Xjit:exclude={package.method*}(count=0,optLevel=0)
		
		FastStringTokenizer st = new FastStringTokenizer(' ');
		
		for(InputSplit split : splits)
		{
			RecordReader<LongWritable,Text> reader = informat.getRecordReader(split, job, Reporter.NULL);
			try
			{
				while(reader.next(key, value))
				{
					st.reset( value.toString() ); //reset tokenizer
					long row = st.nextLong();
				    long col = st.nextLong();
					double lvalue = Double.parseDouble( st.nextToken() );
					
					Cell tmp = new Cell( row, col, lvalue ); 
					
					buffer.addLast( tmp );
					if( buffer.size() > StagingFileUtils.CELL_BUFFER_SIZE ) //periodic flush
					{
						appendCellBufferToStagingArea(fnameStaging, ID, buffer, brlen, bclen);
						buffer.clear();
					}
				}
				
				//final flush
				if( !buffer.isEmpty() )
				{
					appendCellBufferToStagingArea(fnameStaging, ID, buffer, brlen, bclen);
					buffer.clear();
				}
			}
			finally {
				IOUtilFunctions.closeSilently(reader);
			}
		}
	}

	@SuppressWarnings("deprecation")
	private void createBinaryCellStagingFile( String fnameStaging, MatrixObject mo, long ID ) 
		throws IOException, DMLRuntimeException
	{		
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(mo.getFileName());
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
		
		LinkedList<Cell> buffer = new LinkedList<Cell>();
		MatrixIndexes key = new MatrixIndexes();
		MatrixCell value = new MatrixCell();
	
		MatrixCharacteristics mc = mo.getMatrixCharacteristics();
		int brlen = mc.getRowsPerBlock();
		int bclen = mc.getColsPerBlock();
		
		for(Path lpath: IOUtilFunctions.getSequenceFilePaths(fs, path))
		{
			SequenceFile.Reader reader = new SequenceFile.Reader(fs,lpath,job);
			try
			{
				while(reader.next(key, value))
				{
					Cell tmp = new Cell( key.getRowIndex(), key.getColumnIndex(), value.getValue() ); 
	
					buffer.addLast( tmp );
					if( buffer.size() > StagingFileUtils.CELL_BUFFER_SIZE ) //periodic flush
					{
						appendCellBufferToStagingArea(fnameStaging, ID, buffer, brlen, bclen);
						buffer.clear();
					}
				}
				
				//final flush
				if( !buffer.isEmpty() )
				{
					appendCellBufferToStagingArea(fnameStaging, ID, buffer, brlen, bclen);
					buffer.clear();
				}
			}
			finally {
				IOUtilFunctions.closeSilently(reader);
			}
		}
	}

	private void appendCellBufferToStagingArea( String fnameStaging, long ID, LinkedList<Cell> buffer, int brlen, int bclen ) 
		throws DMLRuntimeException, IOException
	{
		HashMap<Long,HashMap<Long,LinkedList<Cell>>> sortedBuffer = new HashMap<Long, HashMap<Long,LinkedList<Cell>>>();
		long brow, bcol, row_offset, col_offset;
		
		for( Cell c : buffer )
		{
			brow = (c.getRow()-1)/brlen + 1;
			bcol = (c.getCol()-1)/bclen + 1;
			row_offset = (brow-1)*brlen + 1;
			col_offset = (bcol-1)*bclen + 1;
			
			c.setRow( c.getRow() - row_offset);
			c.setCol(c.getCol() - col_offset);
			
			if( !sortedBuffer.containsKey(brow) )
				sortedBuffer.put(brow, new HashMap<Long,LinkedList<Cell>>());
			if( !sortedBuffer.get(brow).containsKey(bcol) )
				sortedBuffer.get(brow).put(bcol, new LinkedList<Cell>());
			sortedBuffer.get(brow).get(bcol).addLast(c);
		}	
		
		//write lists of cells to local files
		for( Entry<Long,HashMap<Long,LinkedList<Cell>>> e : sortedBuffer.entrySet() )
		{
			brow = e.getKey();
			for( Entry<Long,LinkedList<Cell>> e2 : e.getValue().entrySet() )
			{
				bcol = e2.getKey();
				String lname = brow+"_"+bcol;
				String dir = fnameStaging+"/"+lname;
				LocalFileUtils.checkAndCreateStagingDir( dir );
				StagingFileUtils.writeCellListToLocal(dir+"/"+ID, e2.getValue());
			}
		}
	}	

	@SuppressWarnings("deprecation")
	private void createBinaryBlockResultFile( String fnameStaging, String fnameStagingCompare, String fnameNew, MatrixFormatMetaData metadata, boolean withCompare ) 
		throws IOException, DMLRuntimeException
	{
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path( fnameNew );	
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
		
		MatrixCharacteristics mc = metadata.getMatrixCharacteristics();
		long rlen = mc.getRows();
		long clen = mc.getCols();
		int brlen = mc.getRowsPerBlock();
		int bclen = mc.getColsPerBlock();
		
		SequenceFile.Writer writer = new SequenceFile.Writer(fs, job, path, MatrixIndexes.class, MatrixBlock.class); //beware ca 50ms
		try
		{
			MatrixIndexes indexes = new MatrixIndexes();
			for(long brow = 1; brow <= (long)Math.ceil(rlen/(double)brlen); brow++)
				for(long bcol = 1; bcol <= (long)Math.ceil(clen/(double)bclen); bcol++)
				{
					File dir = new File(fnameStaging+"/"+brow+"_"+bcol);
					File dir2 = new File(fnameStagingCompare+"/"+brow+"_"+bcol);
					MatrixBlock mb = null;
					
					if( dir.exists() )
					{
						if( withCompare && dir2.exists() ) //WITH COMPARE BLOCK
						{
							//copy only values that are different from the original
							String[] lnames2 = dir2.list();
							if( lnames2.length != 1 ) //there should be exactly 1 compare block
								throw new DMLRuntimeException("Unable to merge results because multiple compare blocks found.");
							mb = LocalFileUtils.readMatrixBlockFromLocal( dir2+"/"+lnames2[0] );
							boolean appendOnly = mb.isInSparseFormat();
							double[][] compare = DataConverter.convertToDoubleMatrix(mb);
							
							String[] lnames = dir.list();
							for( String lname : lnames )
							{
								MatrixBlock tmp = LocalFileUtils.readMatrixBlockFromLocal( dir+"/"+lname );
								mergeWithComp(mb, tmp, compare);
							}
							
							//sort sparse due to append-only
							if( appendOnly )
								mb.sortSparseRows();
							
							//change sparsity if required after 
							mb.examSparsity(); 
						}
						else //WITHOUT COMPARE BLOCK
						{
							//copy all non-zeros from all workers
							String[] lnames = dir.list();
							boolean appendOnly = false;
							for( String lname : lnames )
							{
								if( mb == null )
								{
									mb = LocalFileUtils.readMatrixBlockFromLocal( dir+"/"+lname );
									appendOnly = mb.isInSparseFormat();
								}
								else
								{
									MatrixBlock tmp = LocalFileUtils.readMatrixBlockFromLocal( dir+"/"+lname );
									mergeWithoutComp(mb, tmp, appendOnly);
								}
							}	
							
							//sort sparse due to append-only
							if( appendOnly )
								mb.sortSparseRows();
							
							//change sparsity if required after 
							mb.examSparsity(); 
						}
					}
					else
					{
						//NOTE: whenever runtime does not need all blocks anymore, this can be removed
						int maxRow = (int)(((brow-1)*brlen + brlen < rlen) ? brlen : rlen - (brow-1)*brlen);
						int maxCol = (int)(((bcol-1)*bclen + bclen < clen) ? bclen : clen - (bcol-1)*bclen);
				
						mb = new MatrixBlock(maxRow, maxCol, true);
					}	
					
					//mb.examSparsity(); //done on write anyway and mb not reused
					indexes.setIndexes(brow, bcol);
					writer.append(indexes, mb);
				}	
		}
		finally {
			IOUtilFunctions.closeSilently(writer);
		}
	}

	private void createTextCellResultFile( String fnameStaging, String fnameStagingCompare, String fnameNew, MatrixFormatMetaData metadata, boolean withCompare ) 
		throws IOException, DMLRuntimeException
	{
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path( fnameNew );	
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
		
		MatrixCharacteristics mc = metadata.getMatrixCharacteristics();
		long rlen = mc.getRows();
		long clen = mc.getCols();
		int brlen = mc.getRowsPerBlock();
		int bclen = mc.getColsPerBlock();
				
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(fs.create(path,true)));		
		try
		{
			//for obj reuse and preventing repeated buffer re-allocations
			StringBuilder sb = new StringBuilder();
			
			boolean written=false;
			for(long brow = 1; brow <= (long)Math.ceil(rlen/(double)brlen); brow++)
				for(long bcol = 1; bcol <= (long)Math.ceil(clen/(double)bclen); bcol++)
				{
					File dir = new File(fnameStaging+"/"+brow+"_"+bcol);
					File dir2 = new File(fnameStagingCompare+"/"+brow+"_"+bcol);
					MatrixBlock mb = null;
					
					long row_offset = (brow-1)*brlen + 1;
					long col_offset = (bcol-1)*bclen + 1;
					
					
					if( dir.exists() )
					{
						if( withCompare && dir2.exists() ) //WITH COMPARE BLOCK
						{
							//copy only values that are different from the original
							String[] lnames2 = dir2.list();
							if( lnames2.length != 1 ) //there should be exactly 1 compare block
								throw new DMLRuntimeException("Unable to merge results because multiple compare blocks found.");
							mb = StagingFileUtils.readCellList2BlockFromLocal( dir2+"/"+lnames2[0], brlen, bclen );
							boolean appendOnly = mb.isInSparseFormat();
							double[][] compare = DataConverter.convertToDoubleMatrix(mb);
							
							String[] lnames = dir.list();
							for( String lname : lnames )
							{
								MatrixBlock tmp = StagingFileUtils.readCellList2BlockFromLocal(  dir+"/"+lname, brlen, bclen );
								mergeWithComp(mb, tmp, compare);
							}
							
							//sort sparse and exam sparsity due to append-only
							if( appendOnly )
								mb.sortSparseRows();
							
							//change sparsity if required after 
							mb.examSparsity(); 
						}
						else //WITHOUT COMPARE BLOCK
						{
							//copy all non-zeros from all workers
							String[] lnames = dir.list();
							boolean appendOnly = false;
							for( String lname : lnames )
							{
								if( mb == null )
								{
									mb = StagingFileUtils.readCellList2BlockFromLocal( dir+"/"+lname, brlen, bclen );
									appendOnly = mb.isInSparseFormat();
								}
								else
								{
									MatrixBlock tmp = StagingFileUtils.readCellList2BlockFromLocal(  dir+"/"+lname, brlen, bclen );
									mergeWithoutComp(mb, tmp, appendOnly);
								}
							}	
							
							//sort sparse due to append-only
							if( appendOnly )
								mb.sortSparseRows();
							
							//change sparsity if required after 
							mb.examSparsity(); 
						}
					}

					//write the block to text cell
					if( mb!=null )
					{
						if( mb.isInSparseFormat() )
						{
							Iterator<IJV> iter = mb.getSparseBlockIterator();
							while( iter.hasNext() )
							{
								IJV lcell = iter.next();
								sb.append(row_offset+lcell.getI());
								sb.append(' ');
								sb.append(col_offset+lcell.getJ());
								sb.append(' ');
								sb.append(lcell.getV());
								sb.append('\n');
								out.write( sb.toString() ); 
								sb.setLength(0);
								written = true;
							}							
						}
						else
						{
							for( int i=0; i<brlen; i++ )
								for( int j=0; j<bclen; j++ )
								{
									double lvalue = mb.getValueDenseUnsafe(i, j);
									if( lvalue != 0 ) //for nnz
									{
										sb.append(row_offset+i);
										sb.append(' ');
										sb.append(col_offset+j);
										sb.append(' ');
										sb.append(lvalue);
										sb.append('\n');
										out.write( sb.toString() ); 
										sb.setLength(0);
										written = true;
									}
								}
						}
					}				
				}	
			
			if( !written )
				out.write("1 1 0\n");
		}
		finally {
			IOUtilFunctions.closeSilently(out);
		}
	}

	@SuppressWarnings("deprecation")
	private void createBinaryCellResultFile( String fnameStaging, String fnameStagingCompare, String fnameNew, MatrixFormatMetaData metadata, boolean withCompare ) 
		throws IOException, DMLRuntimeException
	{
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path( fnameNew );	
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
		
		MatrixCharacteristics mc = metadata.getMatrixCharacteristics();
		long rlen = mc.getRows();
		long clen = mc.getCols();
		int brlen = mc.getRowsPerBlock();
		int bclen = mc.getColsPerBlock();
				
		
		MatrixIndexes indexes = new MatrixIndexes(1,1);
		MatrixCell cell = new MatrixCell(0);	
		
		SequenceFile.Writer out = new SequenceFile.Writer(fs, job, path, MatrixIndexes.class, MatrixCell.class); //beware ca 50ms
		try
		{
			boolean written=false;
			for(long brow = 1; brow <= (long)Math.ceil(rlen/(double)brlen); brow++)
				for(long bcol = 1; bcol <= (long)Math.ceil(clen/(double)bclen); bcol++)
				{
					File dir = new File(fnameStaging+"/"+brow+"_"+bcol);
					File dir2 = new File(fnameStagingCompare+"/"+brow+"_"+bcol);
					MatrixBlock mb = null;
					
					long row_offset = (brow-1)*brlen + 1;
					long col_offset = (bcol-1)*bclen + 1;
					
					
					if( dir.exists() )
					{
						if( withCompare && dir2.exists() ) //WITH COMPARE BLOCK
						{
							//copy only values that are different from the original
							String[] lnames2 = dir2.list();
							if( lnames2.length != 1 ) //there should be exactly 1 compare block
								throw new DMLRuntimeException("Unable to merge results because multiple compare blocks found.");
							mb = StagingFileUtils.readCellList2BlockFromLocal( dir2+"/"+lnames2[0], brlen, bclen );
							boolean appendOnly = mb.isInSparseFormat();
							double[][] compare = DataConverter.convertToDoubleMatrix(mb);
							
							String[] lnames = dir.list();
							for( String lname : lnames )
							{
								MatrixBlock tmp = StagingFileUtils.readCellList2BlockFromLocal(  dir+"/"+lname, brlen, bclen );
								mergeWithComp(mb, tmp, compare);
							}
							
							//sort sparse due to append-only
							if( appendOnly )
								mb.sortSparseRows();
							
							//change sparsity if required after 
							mb.examSparsity(); 
						}
						else //WITHOUT COMPARE BLOCK
						{
							//copy all non-zeros from all workers
							String[] lnames = dir.list();
							boolean appendOnly = false;
							for( String lname : lnames )
							{
								if( mb == null )
								{
									mb = StagingFileUtils.readCellList2BlockFromLocal( dir+"/"+lname, brlen, bclen );
									appendOnly = mb.isInSparseFormat();
								}
								else
								{
									MatrixBlock tmp = StagingFileUtils.readCellList2BlockFromLocal(  dir+"/"+lname, brlen, bclen );
									mergeWithoutComp(mb, tmp, appendOnly);
								}
							}	
							
							//sort sparse due to append-only
							if( appendOnly )
								mb.sortSparseRows();
							
							//change sparsity if required after 
							mb.examSparsity(); 
						}
					}
					
					//write the block to binary cell
					if( mb!=null )
					{
						if( mb.isInSparseFormat() )
						{
							Iterator<IJV> iter = mb.getSparseBlockIterator();
							while( iter.hasNext() )
							{
								IJV lcell = iter.next();
								indexes.setIndexes(row_offset+lcell.getI(), col_offset+lcell.getJ());
								cell.setValue(lcell.getV());
								out.append(indexes,cell);
								written = true;
							}
						}
						else
						{
							for( int i=0; i<brlen; i++ )
								for( int j=0; j<bclen; j++ )
								{
									double lvalue = mb.getValueDenseUnsafe(i, j);
									if( lvalue != 0 ) //for nnz
									{
										indexes.setIndexes(row_offset+i, col_offset+j);
										cell.setValue(lvalue);
										out.append(indexes,cell);
										written = true;
									}
								}
						}
					}				
				}	
			
			if( !written )
				out.append(indexes,cell);
		}
		finally {
			IOUtilFunctions.closeSilently(out);
		}
	}

	private void copyAllFiles( String fnameNew, ArrayList<MatrixObject> inMO ) 
		throws CacheException, IOException
	{
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path( fnameNew );
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
		
		//create output dir
		fs.mkdirs(path);
		
		//merge in all input matrix objects
		IDSequence seq = new IDSequence();
		for( MatrixObject in : inMO )
		{			
			LOG.trace("ResultMerge (local, file): Merge input "+in.getVarName()+" (fname="+in.getFileName()+") via file rename.");
			
			//copy over files (just rename file or entire dir)
			Path tmpPath = new Path(in.getFileName());
			String lname = tmpPath.getName();
			fs.rename(tmpPath, new Path(fnameNew+"/"+lname+seq.getNextID()));
		}
	}

}
