package lof;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
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
import org.apache.hadoop.io.NullWritable;
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

import metricspace.IMetric;
import metricspace.IMetricSpace;
import metricspace.MetricObject;
import metricspace.MetricSpaceUtility;
import metricspace.Record;
import sampling.CellStore;
import util.SQConfig;

public class FindKNNSupport {
	/**
	 * default Map class.
	 *
	 * @author Yizhou Yan
	 * @version Dec 31, 2015
	 */

	/** number of object pairs to be computed */
	// static enum Counters {
	// ComputedKnns, UncomputedKnns
	// }
	public static class KNNFinderMapper extends Mapper<LongWritable, Text, IntWritable, Text> {
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

			partition_store = new float[(int) Math.pow(di_numBuckets, num_dims)][num_dims * 2];

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
								for (int j = 0; j < num_dims * 2; j++) {
									partition_store[tempid][j] = Float.parseFloat(splitsStr[j + 1]);
								}
							}
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
								if (items.length == 2) {
									int cellId = Integer.valueOf(items[0]);
									int corePartitionId = Integer.valueOf(items[1].substring(2));
									cell_store[cellId] = new CellStore(cellId, corePartitionId);
								}
							}
						} // end else if
					} // end for (int i = 0; i < stats.length; ++i)
				} // end for (URI path : cacheFiles)

			} catch (IOException ioe) {
				System.err.println("Caught exception while getting cached files");
			}
		}

		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			// Variables
			float[] crds = new float[num_dims];
			String[] splitStr = value.toString().split(SQConfig.sepStrForRecord);
			// parse raw input data into coordinates/crds
			for (int i = 1; i < num_dims + 1; i++) {
				crds[i - 1] = Float.parseFloat(splitStr[i]);
			}
			int cell_id = CellStore.ComputeCellStoreId(crds, num_dims, cell_num, smallRange);
			if (cell_id < 0)
				return;

			int blk_id = cell_store[cell_id].core_partition_id;
			if (blk_id < 0)
				return;
			context.write(new IntWritable(blk_id), value);
		}
	}

	/**
	 * @author yizhouyan
	 *
	 */
	public static class KNNFinderReducer extends Reducer<IntWritable, Text, LongWritable, Text> {
		/**
		 * The dimension of data (set by user, now only support dimension of 2,
		 * if change to 3 or more, has to change some codes)
		 */
		private static int num_dims = 2;
		/** The domains. (set by user) */
		private static float[] domains;
		/**
		 * block list, which saves each block's info including start & end
		 * positions on each dimension. print for speed up "mapping"
		 */
		private static float[][] partition_store;
		/**
		 * Number of desired partitions in each dimension (set by user), for
		 * Data Driven partition
		 */
		private static int di_numBuckets;
		private static int K;
		private IMetricSpace metricSpace = null;
		private IMetric metric = null;

		private MultipleOutputs mos;

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
			mos = new MultipleOutputs(context);
			Configuration conf = context.getConfiguration();
			readMetricAndMetricSpace(conf);
			/** get configuration from file */
			num_dims = conf.getInt(SQConfig.strDimExpression, 2);
			domains = new float[2];
			domains[0] = conf.getFloat(SQConfig.strDomainMin, 0.0f);
			domains[1] = conf.getFloat(SQConfig.strDomainMax, 10001.0f);
			di_numBuckets = conf.getInt(SQConfig.strNumOfPartitions, 2);
			partition_store = new float[(int) Math.pow(di_numBuckets, num_dims)][num_dims * 2];

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
								for (int j = 0; j < num_dims * 2; j++) {
									partition_store[tempid][j] = Float.parseFloat(splitsStr[j + 1]);
								}
							}
						}
					} // end for (int i = 0; i < stats.length; ++i)
				} // end for (URI path : cacheFiles)

			} catch (IOException ioe) {
				System.err.println("Caught exception while getting cached files");
			}
		}

		private MetricObject parseObject(int key, String strInput) {
			int partition_id = key;
			int offset = 0;
			Object obj = metricSpace.readObject(strInput.substring(offset, strInput.length()), num_dims);
			return new MetricObject(partition_id, obj);
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
			float[] partitionExpand = new float[num_dims * 2];
			for (int i = 0; i < num_dims * 2; i++)
				partitionExpand[i] = 0.0f;

			ArrayList<MetricObject> sortedData = new ArrayList<MetricObject>();
			for (Text value : values) {
				MetricObject mo = parseObject(key.get(), value.toString());
				sortedData.add(mo);
			}
			String centralPivotStr = "0";
			for (int j = 0; j < num_dims; j++) {
				centralPivotStr = centralPivotStr + ","
						+ (partition_store[key.get()][j * 2] + partition_store[key.get()][j * 2 + 1]) / 2.0;
			}

			Object cPivot = metricSpace.readObject(centralPivotStr, num_dims);
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
			 * "Entry: " + entry.obj.toString()); }
			 */
			long begin = System.currentTimeMillis();

			if (sortedData.size() < K) {
				System.err.println("Contains less than K points in one partition....");
			}
			String outputKDistPath = context.getConfiguration().get(SQConfig.strKdistanceOutput);
			for (int i = 0; i < sortedData.size(); i++) {
				MetricObject o_S = sortedData.get(i);
				// find knns for single object within the partition
				o_S = findKNNForSingleObject(o_S, i, sortedData);
				// bound supporting area
				float[] currentBound = boundSupportingArea(o_S);
				if (o_S.getType() == 'F')
					// set partition expand distance by points in that partition
					partitionExpand = maxOfTwoFloatArray(partitionExpand, currentBound);
				// output data point
				outputMultipleTypeData(o_S, outputKDistPath, context);
				// if (o_S.getType() == 'T')
				// context.getCounter(Counters.ComputedKnns).increment(1);
				// else if (o_S.getType() == 'F')
				// context.getCounter(Counters.UncomputedKnns).increment(1);
			}
			// output partition plan
			int curPartitionId = sortedData.get(0).getPartition_id();
			String outputPPPath = context.getConfiguration().get(SQConfig.strKnnPartitionPlan);
			String str = "";
			str += curPartitionId + SQConfig.sepStrForRecord;
			for (int i = 0; i < num_dims * 2; i++) {
				str += partition_store[curPartitionId][i] + SQConfig.sepStrForRecord;
			}
			for (int i = 0; i < num_dims * 2; i++) {
				str += partitionExpand[i] + SQConfig.sepStrForRecord;
			}
			str = str.substring(0, str.length() - 1);
			mos.write(NullWritable.get(), new Text(str), outputPPPath + "/pp" + context.getTaskAttemptID());
			long end = System.currentTimeMillis();
			long second = (end - begin) / 1000;
			System.err.println("computation time " + " takes " + second + " seconds");
		}

		public static float[] maxOfTwoFloatArray(float[] x, float[] y) {
			float[] newArray = new float[num_dims * 2];
			for (int i = 0; i < x.length; i++) {
				newArray[i] = Math.max(x[i], y[i]);
			}
			return newArray;
		}

		/**
		 * output different types of data points in multiple files
		 * 
		 * @param context
		 * @param o_R
		 * @throws InterruptedException
		 * @throws IOException
		 */
		public void outputMultipleTypeData(MetricObject o_R, String outputKDistPath, Context context)
				throws IOException, InterruptedException {
			// output format key:nid value: point value, partition id,
			// k-distance, (KNN's nid and dist),tag
			LongWritable outputKey = new LongWritable();
			Text outputValue = new Text();

			String line = "";
			line = line + o_R.getPartition_id() + SQConfig.sepStrForRecord + o_R.getKdist() + SQConfig.sepStrForRecord;
			for (Map.Entry<Long, Float> entry : o_R.getKnnInDetail().entrySet()) {
				long keyMap = entry.getKey();
				float valueMap = entry.getValue();
				line = line + keyMap + SQConfig.sepStrForIDDist + valueMap + SQConfig.sepStrForRecord;
			}
			line = line + o_R.getType();
			outputKey.set(((Record) o_R.getObj()).getRId());
			outputValue.set(((Record) o_R.getObj()).dimToString() + SQConfig.sepStrForRecord + line);
			mos.write(outputKey, outputValue, outputKDistPath + "/kdist" + context.getTaskAttemptID());
			/*
			 * if(o_R.getType()=='T') mos.write(outputKey,outputValue,
			 * outputKDistPath + "/yes/kdist"); else if(o_R.getType() == 'F')
			 * mos.write(outputKey,outputValue, outputKDistPath + "/not/kdist");
			 */
			// System.out.println("Key: "+ outputKey +" Value: "+outputValue);
			// context.write(outputKey, outputValue);
		}

		/**
		 * 
		 * @param o_R
		 *            MetricObject with kdistance and knns
		 * @return MetricObject with tag (T: true kNN, already find kNNs within
		 *         the partition N: false kNN, kNNs might in other partitions)
		 */
		private float[] boundSupportingArea(MetricObject o_R) {
			float[] boundForCurPoint = new float[num_dims * 2];
			for (int i = 0; i < num_dims * 2; i++)
				boundForCurPoint[i] = 0.0f;

			Record currentPoint = (Record) o_R.getObj();
			float[] currentPointCoor = currentPoint.getValue();
			float currentKDist = o_R.getKdist();
			boolean tag = true;
			for (int i = 0; i < num_dims; i++) {
				float minCurrent = Math.max(domains[0], currentPointCoor[i] - currentKDist);
				float maxCurrent = Math.min(domains[1] - Float.MIN_VALUE, currentPointCoor[i] + currentKDist);
				if (minCurrent < partition_store[o_R.getPartition_id()][2 * i]) {
					tag = false;
					boundForCurPoint[2 * i] = Math.max(boundForCurPoint[2 * i],
							Math.abs(partition_store[o_R.getPartition_id()][2 * i] - minCurrent));
				}
				if (maxCurrent > partition_store[o_R.getPartition_id()][2 * i + 1]) {
					tag = false;
					boundForCurPoint[2 * i + 1] = Math.max(boundForCurPoint[2 * i + 1],
							Math.abs(partition_store[o_R.getPartition_id()][2 * i + 1] - maxCurrent));
				} else if (maxCurrent == partition_store[o_R.getPartition_id()][2 * i + 1]) {
					tag = false;
					boundForCurPoint[2 * i + 1] = Math.max(Float.MIN_VALUE, boundForCurPoint[2 * i + 1]);
				}
			}
			if (tag == false)
				o_R.setType('F');
			else
				o_R.setType('T');
			return boundForCurPoint;
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

			float theta = Float.POSITIVE_INFINITY;
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
					if (pq.size() < K) {
						pq.insert(metricSpace.getID(o_S.getObj()), dist);
						theta = pq.getPriority();
					} else if (dist < theta) {
						pq.pop();
						pq.insert(metricSpace.getID(o_S.getObj()), dist);
						theta = pq.getPriority();
					}
					inc_current += 1;
					i = Math.abs(o_R.getDistToPivot() - o_S.getDistToPivot());
				} else {
					MetricObject o_S = sortedData.get(dec_current);
					dist = metric.dist(o_R.getObj(), o_S.getObj());
					if (pq.size() < K) {
						pq.insert(metricSpace.getID(o_S.getObj()), dist);
						theta = pq.getPriority();
					} else if (dist < theta) {
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
			while (pq.size() > 0) {
				o_R.getKnnInDetail().put(pq.getValue(), pq.getPriority());
				pq.pop();
			}
			return o_R;
		}

		public void cleanup(Context context) throws IOException, InterruptedException {
			mos.close();
		}
	}

	public void run(String[] args) throws Exception {
		Configuration conf = new Configuration();
		conf.addResource(new Path("/usr/local/Cellar/hadoop/etc/hadoop/core-site.xml"));
		conf.addResource(new Path("/usr/local/Cellar/hadoop/etc/hadoop/hdfs-site.xml"));
		new GenericOptionsParser(conf, args).getRemainingArgs();
		/** set job parameter */
		Job job = Job.getInstance(conf, "DDLOF-calculate Kdistance 1st job");

		job.setJarByClass(FindKNNSupport.class);
		job.setMapperClass(KNNFinderMapper.class);

		/** set multiple output path */
		MultipleOutputs.addNamedOutput(job, "partitionplan", TextOutputFormat.class, NullWritable.class, Text.class);
		MultipleOutputs.addNamedOutput(job, "kdistance", TextOutputFormat.class, LongWritable.class, Text.class);

		job.setReducerClass(KNNFinderReducer.class);
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(Text.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(Text.class);
		job.setNumReduceTasks(conf.getInt(SQConfig.strNumOfReducers, 1));
		// job.setNumReduceTasks(0);
		String strFSName = conf.get("fs.default.name");
		FileInputFormat.addInputPath(job, new Path(conf.get(SQConfig.dataset)));
		FileSystem fs = FileSystem.get(conf);
		fs.delete(new Path(conf.get(SQConfig.strKnnSummaryOutput)), true);
		FileOutputFormat.setOutputPath(job, new Path(conf.get(SQConfig.strKnnSummaryOutput)));
		job.addCacheArchive(new URI(strFSName + conf.get(SQConfig.strPartitionPlanOutput)));
		job.addCacheArchive(new URI(strFSName + conf.get(SQConfig.strCellsOutput)));

		/** print job parameter */
		System.err.println("# of dim: " + conf.getInt(SQConfig.strDimExpression, 10));
		long begin = System.currentTimeMillis();
		job.waitForCompletion(true);
		long end = System.currentTimeMillis();
		long second = (end - begin) / 1000;
		System.err.println(job.getJobName() + " takes " + second + " seconds");
	}

	public static void main(String[] args) throws Exception {
		FindKNNSupport findKnnAndSupporting = new FindKNNSupport();
		findKnnAndSupporting.run(args);
	}
}
