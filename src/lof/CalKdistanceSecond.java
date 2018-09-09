package lof;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import lof.PriorityQueue;

import lof.FindKNNSupport.KNNFinderReducer;
import metricspace.IMetric;
import metricspace.IMetricSpace;
import metricspace.MetricObject;
import metricspace.MetricSpaceUtility;
import metricspace.Record;
import sampling.CellStore;
import util.SQConfig;

public class CalKdistanceSecond {
	/**
	 * default Map class.
	 *
	 * @author Yizhou Yan
	 * @version Dec 31, 2015
	 */

	public static class CalKdistSecondMapper extends Mapper<LongWritable, Text, IntWritable, Text> {
		/**
		 * The dimension of data (set by user, now only support dimension of 2,
		 * if change to 3 or more, has to change some codes)
		 */
		private static int num_dims = 2;
		/**
		 * number of small cells per dimension: when come with a node, map to a
		 * range (divide the domain into small_cell_num_per_dim) (set by user)
		 */
		public static int cell_num = 501;

		/** The domains. (set by user) */
		private static float[] domains;
		/** size of each small buckets */
		private static int smallRange;
		/**
		 * block list, which saves each block's info including start & end
		 * positions on each dimension. print for speed up "mapping"
		 */
		private static float[][] partition_store;
		/** save each small buckets. in order to speed up mapping process */
		private static CellStore[] cell_store;
		/**
		 * Number of desired partitions in each dimension (set by user), for
		 * Data Driven partition
		 */
		private static int di_numBuckets;

		private static int K;

		protected void setup(Context context) throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
			/** get configuration from file */
			num_dims = conf.getInt(SQConfig.strDimExpression, 2);
			cell_num = conf.getInt(SQConfig.strNumOfSmallCells, 501);
			domains = new float[2];
			domains[0] = conf.getFloat(SQConfig.strDomainMin, 0.0f);
			domains[1] = conf.getFloat(SQConfig.strDomainMax, 10001.0f);
			smallRange = (int) Math.ceil((domains[1] - domains[0]) / cell_num);
			cell_store = new CellStore[(int) Math.pow(cell_num, num_dims)];
			di_numBuckets = conf.getInt(SQConfig.strNumOfPartitions, 2);

			partition_store = new float[(int) Math.pow(di_numBuckets, num_dims)][num_dims * 4];
			K = Integer.valueOf(conf.get(SQConfig.strK, "1"));
			/** parse files in the cache */
			try {
				URI[] cacheFiles = context.getCacheArchives();

				if (cacheFiles == null || cacheFiles.length < 2) {
					System.out.println("not enough cache files");
					return;
				}
				for (URI path : cacheFiles) {
					String filename = path.toString();
					FileSystem fs = FileSystem.get(conf);

					FileStatus[] stats = fs.listStatus(new Path(filename));
					for (int i = 0; i < stats.length; ++i) {
						if (!stats[i].isDirectory() && stats[i].getPath().toString().contains("pp")) {
							System.out.println("Reading partition plan from " + stats[i].getPath().toString());
							FSDataInputStream currentStream;
							BufferedReader currentReader;
							currentStream = fs.open(stats[i].getPath());
							currentReader = new BufferedReader(new InputStreamReader(currentStream));
							String line;
							while ((line = currentReader.readLine()) != null) {
								/** parse line */

								String[] splitsStr = line.split(SQConfig.sepStrForRecord);
								int tempid = Integer.parseInt(splitsStr[0]);
								for (int j = 1; j < num_dims * 4 + 1; j++) {
									partition_store[tempid][j - 1] = Float.parseFloat(splitsStr[j]);
									// System.out.print(partition_store[tempid][j
									// - 1] + ",");
								}
								// System.out.println();
							}
							currentReader.close();
							currentStream.close();
						} else if (!stats[i].isDirectory() && stats[i].getPath().toString().contains("part")) {
							System.out.println("Reading cells for partitions from " + stats[i].getPath().toString());
							FSDataInputStream currentStream;
							BufferedReader currentReader;
							currentStream = fs.open(stats[i].getPath());
							currentReader = new BufferedReader(new InputStreamReader(currentStream));
							String line;
							while ((line = currentReader.readLine()) != null) {
								/** parse line */
								String[] items = line.split(SQConfig.sepStrForRecord);
								if (items.length == 3) {
									int cellId = Integer.parseInt(items[0]);
									int corePartitionId = Integer.valueOf(items[1].substring(2));
									cell_store[cellId] = new CellStore(cellId, corePartitionId);
									if (items[2].length() > 1) { // has support
																	// cells
										String[] splitStr = items[2].substring(2, items[2].length())
												.split(SQConfig.sepSplitForIDDist);
										for (int j = 0; j < splitStr.length; j++) {
											cell_store[cellId].support_partition_id.add(Integer.valueOf(splitStr[j]));
										}
									}
									// System.out.println(cell_store[cellId].printCellStoreWithSupport());
								}
							}
							currentReader.close();
							currentStream.close();
						} // end else if
					} // end for (int i = 0; i < stats.length; ++i)
				} // end for (URI path : cacheFiles)

			} catch (IOException ioe) {
				System.err.println("Caught exception while getting cached files");
			}
			System.out.println("End Setting up");
		}

		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			// Variables
			// input format key:nid value: point value, partition id,
			// k-distance, (KNN's nid and dist),tag
			String inputStr = value.toString();
			if (inputStr.split(SQConfig.sepStrForKeyValue).length < 2) {
				System.out.println("Error Output: " + inputStr);
			} else {
				String[] inputStrSplits = inputStr.split(SQConfig.sepStrForKeyValue)[1].split(SQConfig.sepStrForRecord);
				if (inputStrSplits.length < 3 + num_dims + K) {
					return;
				}
				long pointId = Long.valueOf(inputStr.split(SQConfig.sepStrForKeyValue)[0]);
				float[] crds = new float[num_dims];
				// parse raw input data into coordinates/crds
				for (int i = 0; i < num_dims; i++) {
					crds[i] = Float.parseFloat(inputStrSplits[i]);
				}
				// core partition id
				int corePartitionId = Integer.valueOf(inputStrSplits[num_dims]);

				// k-distance saved
				float curKdist = (inputStrSplits[num_dims + 1].equals("")) ? 0
						: Float.valueOf(inputStrSplits[num_dims + 1]);

				// knns
				String curKnns = "";
				for (int i = num_dims + 2; i < inputStrSplits.length - 1; i++) {
					curKnns = curKnns + inputStrSplits[i] + SQConfig.sepStrForRecord;
				}
				if (curKnns.length() > 0)
					curKnns = curKnns.substring(0, curKnns.length() - 1);

				// tag
				char curTag = inputStrSplits[inputStrSplits.length - 1].charAt(0);

				// find which cell the point in

				int cellStoreId = CellStore.ComputeCellStoreId(crds, num_dims, cell_num, smallRange);
				// System.out.println("CellStore ID: " + cellStoreId);

				// build up partitions to check and save to a hash set (by core
				// area
				// and support area of the cell)
				Set<Integer> partitionToCheck = new HashSet<Integer>();
				// partitionToCheck.add(cell_store[cellStoreId].core_partition_id);
				for (Iterator itr = cell_store[cellStoreId].support_partition_id.iterator(); itr.hasNext();) {
					int keyiter = (Integer) itr.next();
					partitionToCheck.add(keyiter);
				}

				String whoseSupport = "";
				// traverse each block to find belonged regular or extended
				// block
				for (Iterator iter = partitionToCheck.iterator(); iter.hasNext();) {
					int blk_id = (Integer) iter.next();
					if (blk_id < 0) {
						System.out.println("Block id: " + blk_id);
						continue;
					}
					// for(int blk_id=0;blk_id<markEndPoints(crds[0]);blk_id++)
					// {
					int belong = 0; // indicate whether the point belongs, 0
									// ->neither; 2-> extend
					// traverse block's start & end positions in each dimension
					for (int i = 0; i < num_dims; i++) {
						if (crds[i] < partition_store[blk_id][2 * i + 1]
								+ partition_store[blk_id][2 * num_dims + 2 * i + 1]
								&& crds[i] >= partition_store[blk_id][2 * i]
										- partition_store[blk_id][2 * num_dims + 2 * i]) {
							belong = 2;
						} else {
							belong = 0;
							break;
						}
					} // end for(int i=0;i<numDims;i++)

					// output block key and data value
					if (belong == 2) { // support area data
						// output to support area with a tag 'S'
						String str = "";
						str = str + pointId + SQConfig.sepStrForRecord;
						for (int i = 0; i < num_dims; i++)
							str = str + crds[i] + SQConfig.sepStrForRecord;

						context.write(new IntWritable(blk_id), new Text(str + 'S'));
						// save information to whoseSupport
						whoseSupport = whoseSupport + blk_id + SQConfig.sepStrForIDDist;
					} // end if
				} // end for(int blk_id=0;blk_id<blocklist.length;blk_id++)
				if (whoseSupport.length() > 0)
					whoseSupport = whoseSupport.substring(0, whoseSupport.length() - 1);

				// output core area
				// format key : core partition id value: nid,node
				// information,kdistance, knns, tag
				String str = "";
				str = str + pointId + SQConfig.sepStrForRecord;
				for (int i = 0; i < num_dims; i++)
					str = str + crds[i] + SQConfig.sepStrForRecord;
				context.write(new IntWritable(corePartitionId), new Text(str + curKdist + SQConfig.sepStrForRecord
						+ curKnns + SQConfig.sepStrForRecord + curTag + SQConfig.sepStrForRecord + whoseSupport));
			}
		}// end map function
	} // end map class

	public static class CalKdistSecondReducer extends Reducer<IntWritable, Text, LongWritable, Text> {
		/**
		 * The dimension of data (set by user, now only support dimension of 2,
		 * if change to 3 or more, has to change some codes)
		 */
		private static int num_dims = 2;
		private static int K;
		private IMetricSpace metricSpace = null;
		private IMetric metric = null;

		/**
		 * get MetricSpace and metric from configuration
		 * 
		 * @param conf
		 * @throws IOException
		 */
		private void readMetricAndMetricSpace(Configuration conf) throws IOException {
			try {
				metricSpace = MetricSpaceUtility.getMetricSpace(conf.get(SQConfig.strMetricSpace));
				metric = MetricSpaceUtility.getMetric(conf.get(SQConfig.strMetric));
				metricSpace.setMetric(metric);
			} catch (InstantiationException e) {
				throw new IOException("InstantiationException");
			} catch (IllegalAccessException e) {
				e.printStackTrace();
				throw new IOException("IllegalAccessException");
			} catch (ClassNotFoundException e) {
				e.printStackTrace();
				throw new IOException("ClassNotFoundException");
			}
		}

		public void setup(Context context) throws IOException {
			Configuration conf = context.getConfiguration();
			readMetricAndMetricSpace(conf);
			/** get configuration from file */
			num_dims = conf.getInt(SQConfig.strDimExpression, 2);
			K = Integer.valueOf(conf.get(SQConfig.strK, "1"));
		}

		/**
		 * parse objects in supporting area key: partition id value: point id,
		 * point information(2-d), tag(S)
		 * 
		 * @param key:
		 *            partition id
		 * @param strInput:
		 *            point id, point information(2-d), tag(S)
		 * @return
		 */
		private MetricObject parseSupportObject(int key, String strInput) {
			int partition_id = key;
			int offset = 0;
			Object obj = metricSpace.readObject(strInput.substring(offset, strInput.length() - 2), num_dims);
			char curTag = strInput.charAt(strInput.length() - 1);
			return new MetricObject(partition_id, obj, curTag);
		}

		/**
		 * parse objects in core area key: partition id value: point id, point
		 * information(2-d), k-distance, knns, tag(S), whoseSupport
		 * 
		 * @param key
		 * @param strInput
		 * @return
		 */
		private MetricObject parseCoreObject(int key, String strInput) {
			String[] splitStrInput = strInput.split(SQConfig.sepStrForRecord);
			int partition_id = key;
			int offset = 0;
			int lengthOfObjects = splitStrInput[0].length();
			for (int i = 1; i <= num_dims; i++) {
				lengthOfObjects += splitStrInput[i].length() + 1;
			}
			Object obj = metricSpace.readObject(strInput.substring(offset, lengthOfObjects), num_dims);
			float curKdist = Float.parseFloat(splitStrInput[num_dims + 1]);
			offset = lengthOfObjects + splitStrInput[num_dims + 1].length() + 2;
			int endoffset = strInput.indexOf("F");
			String[] subSplits = strInput.substring(offset, endoffset - 1).split(SQConfig.sepStrForRecord);
			
			Map<Long, Float> knnInDetail = new HashMap<Long, Float>();
			for (int i = 0; i < subSplits.length; i++) {
				long knnid = Long.parseLong(subSplits[i].split(SQConfig.sepSplitForIDDist)[0]);
				float knndist = Float.parseFloat(subSplits[i].split(SQConfig.sepSplitForIDDist)[1]);
				knnInDetail.put(knnid, knndist);
			}
			char curTag = 'F';
			String whoseSupport = strInput.substring(strInput.indexOf("F") + 2, strInput.length());
			return new MetricObject(partition_id, obj, curKdist, knnInDetail, curTag, whoseSupport);
		}

		/**
		 * default Reduce class.
		 * 
		 * @author Yizhou Yan
		 * @version Dec 31, 2015
		 * @throws InterruptedException
		 */

		public void reduce(IntWritable key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {
			ArrayList<MetricObject> sortedData = new ArrayList<MetricObject>();
			int countSupporting = 0;
			boolean moreSupporting = true;
			for (Text value : values) {
				String[] splitStrInput = value.toString().split(SQConfig.sepStrForRecord);
				if ((splitStrInput.length == (2 + num_dims))) {
					MetricObject mo = parseSupportObject(key.get(), value.toString());
					if(moreSupporting)
						sortedData.add(mo);
					countSupporting++;
					if (countSupporting >= 8000000) {
						moreSupporting = false;
					}
				} else if (value.toString().contains("F")) {
					MetricObject mo = parseCoreObject(key.get(), value.toString());
					sortedData.add(mo);
				} else if (value.toString().contains("T")) {
					// output those already know extract knns
					int offset = 0;
					float curKdist = Float.parseFloat(splitStrInput[num_dims + 1]);
					int lengthOfObjects = splitStrInput[0].length();
					for (int i = 1; i <= num_dims; i++) {
						lengthOfObjects += splitStrInput[i].length() + 1;
					}
					offset = lengthOfObjects + splitStrInput[num_dims + 1].length() + 2;
					int endoffset = value.toString().indexOf("T");
					String knnsDetail = value.toString().substring(offset, endoffset - 1);
					String whoseSupport = value.toString().substring(value.toString().indexOf("T") + 2,
							value.toString().length());
					context.write(new LongWritable(Long.parseLong(splitStrInput[0])),
							new Text(key.toString() + SQConfig.sepStrForRecord + curKdist + SQConfig.sepStrForRecord
									+ whoseSupport + SQConfig.sepStrForRecord + knnsDetail));
				}
			} // end for collect data
			if(moreSupporting == false){
				System.out.println("Partitions larger than 800w:" + countSupporting);
			}
			// if no data left in this partition, return
			if (sortedData.size() == 0)
				return;
			// else select the first one as a pivot
			Object cPivot = sortedData.get(0).getObj();
			// calculate distance to the pivot (in order to build the index)
			for (int i = 0; i < sortedData.size(); i++) {
				sortedData.get(i).setDistToPivot(metric.dist(cPivot, sortedData.get(i).getObj()));
			}

			Collections.sort(sortedData, new Comparator<MetricObject>() {
				public int compare(MetricObject map1, MetricObject map2) {
					if (map2.getDistToPivot() > map1.getDistToPivot())
						return 1;
					else if (map2.getDistToPivot() < map1.getDistToPivot())
						return -1;
					else
						return 0;
				}
			});
			/*
			 * for (MetricObject entry : sortedData) { System.out.println(
			 * "Entry: " + ((Record)entry.getObj()).toString()); }
			 */
			long begin = System.currentTimeMillis();
			for (int i = 0; i < sortedData.size(); i++) {
				MetricObject o_S = sortedData.get(i);
				// find knns for single object within the partition
				if (o_S.getType() == 'F') {
					o_S = findKNNForSingleObject(o_S, i, sortedData);
					// output data point
					// output format key:nid
					// value: partition id, point value, k-distance, (KNN's nid
					// and dist),tag, whoseSupport
					LongWritable outputKey = new LongWritable();
					Text outputValue = new Text();
					String line = "";
					line = line + o_S.getPartition_id() + SQConfig.sepStrForRecord + o_S.getKdist()
							+ SQConfig.sepStrForRecord + o_S.getWhoseSupport() + SQConfig.sepStrForRecord;
					for (Map.Entry<Long, Float> entry : o_S.getKnnInDetail().entrySet()) {
						long keyMap = entry.getKey();
						float valueMap = entry.getValue();
						line = line + keyMap + SQConfig.sepStrForIDDist + valueMap + SQConfig.sepStrForRecord;
					}
					line = line.substring(0, line.length() - 1);
					outputKey.set(((Record) o_S.getObj()).getRId());
					outputValue.set(line);
					context.write(outputKey, outputValue);
				}
			}
			long end = System.currentTimeMillis();
			long second = (end - begin) / 1000;
			System.err.println("computation time " + " takes " + second + " seconds");
		}

		/**
		 * find kNN using pivot based index
		 * 
		 * @return MetricObject with kdistance and knns
		 * @throws InterruptedException
		 */
		private MetricObject findKNNForSingleObject(MetricObject o_R, int currentIndex,
				ArrayList<MetricObject> sortedData) throws IOException, InterruptedException {
			float dist;
			PriorityQueue pq = new PriorityQueue(PriorityQueue.SORT_ORDER_DESCENDING);
			// load original knns
			float theta = Float.POSITIVE_INFINITY;
			for (Map.Entry<Long, Float> entry : o_R.getKnnInDetail().entrySet()) {
				long keyMap = entry.getKey();
				float valueMap = entry.getValue();
				// System.out.println("For data "+ o_R.getObj().toString() +"
				// knns: "+ keyMap + ","+ valueMap);
				pq.insert(keyMap, valueMap);
				theta = pq.getPriority();
			}

			boolean kNNfound = false;
			int inc_current = currentIndex + 1;
			int dec_current = currentIndex - 1;
			float i = 0, j = 0; // i---increase j---decrease
			while ((!kNNfound) && ((inc_current < sortedData.size()) || (dec_current >= 0))) {
				// System.out.println("increase: "+ inc_current+"; decrease:
				// "+dec_current);
				if ((inc_current > sortedData.size() - 1) && (dec_current < 0))
					break;
				if (inc_current > sortedData.size() - 1)
					i = Float.MAX_VALUE;
				if (dec_current < 0)
					j = Float.MAX_VALUE;
				if (i <= j) {
					MetricObject o_S = sortedData.get(inc_current);

					dist = metric.dist(o_R.getObj(), o_S.getObj());
					if ((pq.size() < K) && (o_S.getType() == 'S')) {
						pq.insert(metricSpace.getID(o_S.getObj()), dist);
						theta = pq.getPriority();
					} else if ((dist < theta) && (o_S.getType() == 'S')) {
						pq.pop();
						pq.insert(metricSpace.getID(o_S.getObj()), dist);
						theta = pq.getPriority();
					}
					inc_current += 1;
					i = Math.abs(o_R.getDistToPivot() - o_S.getDistToPivot());
				} else {
					MetricObject o_S = sortedData.get(dec_current);
					dist = metric.dist(o_R.getObj(), o_S.getObj());
					if ((pq.size() < K) && (o_S.getType() == 'S')) {
						pq.insert(metricSpace.getID(o_S.getObj()), dist);
						theta = pq.getPriority();
					} else if ((dist < theta) && (o_S.getType() == 'S')) {
						pq.pop();
						pq.insert(metricSpace.getID(o_S.getObj()), dist);
						theta = pq.getPriority();
					}
					dec_current -= 1;
					j = Math.abs(o_R.getDistToPivot() - o_S.getDistToPivot());
				}
				// System.out.println(pq.getPriority()+","+i+","+j);
				if (i > pq.getPriority() && j > pq.getPriority() && (pq.size() == K))
					kNNfound = true;
			}
			o_R.setKdist(pq.getPriority());
			o_R.getKnnInDetail().clear();
			while (pq.size() > 0) {
				o_R.getKnnInDetail().put(pq.getValue(), pq.getPriority());
				// System.out.println("knns: "+ pq.getValue() + ","+
				// pq.getPriority());
				pq.pop();
			}
			return o_R;
		}
	}

	public void run(String[] args) throws Exception {
		Configuration conf = new Configuration();
		conf.addResource(new Path("/usr/local/Cellar/hadoop/etc/hadoop/core-site.xml"));
		conf.addResource(new Path("/usr/local/Cellar/hadoop/etc/hadoop/hdfs-site.xml"));
		new GenericOptionsParser(conf, args).getRemainingArgs();
		/** set job parameter */
		Job job = Job.getInstance(conf, "DDLOF-calculate kdistance 2nd job");

		job.setJarByClass(CalKdistanceSecond.class);
		job.setMapperClass(CalKdistSecondMapper.class);

		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(Text.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(Text.class);
		job.setReducerClass(CalKdistSecondReducer.class);
		 job.setNumReduceTasks(conf.getInt(SQConfig.strNumOfReducers, 1));
//		job.setNumReduceTasks(0);

		String strFSName = conf.get("fs.default.name");
		FileInputFormat.addInputPath(job, new Path(conf.get(SQConfig.strKdistanceOutput)));
		FileSystem fs = FileSystem.get(conf);
		fs.delete(new Path(conf.get(SQConfig.strKdistFinalOutput)), true);
		FileOutputFormat.setOutputPath(job, new Path(conf.get(SQConfig.strKdistFinalOutput)));
		job.addCacheArchive(new URI(strFSName + conf.get(SQConfig.strKnnPartitionPlan)));
		job.addCacheArchive(new URI(strFSName + conf.get(SQConfig.strKnnCellsOutput)));

		/** print job parameter */
		System.err.println("# of dim: " + conf.getInt(SQConfig.strDimExpression, 10));
		long begin = System.currentTimeMillis();
		job.waitForCompletion(true);
		long end = System.currentTimeMillis();
		long second = (end - begin) / 1000;
		System.err.println(job.getJobName() + " takes " + second + " seconds");
	}

	public static void main(String[] args) throws Exception {
		CalKdistanceSecond findKnnAndSupporting = new CalKdistanceSecond();
		findKnnAndSupporting.run(args);
	}
}
