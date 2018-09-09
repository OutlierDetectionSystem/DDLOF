package sampling;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.util.ToolRunner;

import util.SQConfig;

/**
 * DDrivePartition class contains map and reduce functions for Data Driven
 * Partitioning.
 *
 * @author Yizhou Yan
 * @version Dec 31, 2015
 */
public class DataDrivenPartition {
	/**
	 * default Map class.
	 *
	 * @author Yizhou Yan
	 * @version Dec 31, 2015
	 */

	public static class DDMapper extends Mapper<LongWritable, Text, IntWritable, Text> {

		/** incrementing index of divisions generated in mapper */
		public static int increIndex = 0;

		/** number of divisions where data is divided into (set by user) */
		private int denominator = 100;

		/** number of object pairs to be computed */
		static enum Counters {
			MapCount
		}

		protected void setup(Context context) throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
			denominator = conf.getInt(SQConfig.strSamplingPercentage, 100);
		}

		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			int id = increIndex % denominator;
			increIndex++;

			Text dat = new Text(value.toString());
			IntWritable key_id = new IntWritable(id);
			if (id == 0) {
				context.write(key_id, dat);
				context.getCounter(Counters.MapCount).increment(1);
			}
		}
	}

	/**
	 * @author yizhouyan
	 *
	 */
	public static class DDReducer extends Reducer<IntWritable, Text, NullWritable, Text> {
		private MultipleOutputs<NullWritable, Text> mos;
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
		 * Number of desired partitions in each dimension (set by user), for
		 * Data Driven partition
		 */
		private static int di_numBuckets;
		/**
		 * block list, which saves each block's info including start & end
		 * positions on each dimension. print for speed up "mapping"
		 */
		private static float[][] partition_store;
		/** the total usage of the block list (partition store) */
		private static int indexForPartitionStore = 0;
		/** in order to build hash to speed up mapping process */
		public static Hashtable<Double, StartAndEnd> start_end_points;

		private static int[] partition_size;

		public static class StartAndEnd {
			public int start_point;
			public int end_point;

			public StartAndEnd(int start_point, int end_point) {
				this.start_point = start_point;
				this.end_point = end_point;
			}
		}

		/**
		 * sort chi squares by dimension
		 * 
		 * @param chisquares
		 *            an array of chisquares
		 * @return a sorted dimension
		 */
		public int[] sortDim(double[] chisquares) {
			int len = chisquares.length;
			int[] sorted_dim = new int[len];
			for (int i = 0; i < len; i++) {
				sorted_dim[i] = i;
			}
			for (int i = 0; i < len; i++) {
				for (int j = i + 1; j < len; j++) {
					double temp;
					int tempindex;
					if (chisquares[i] > chisquares[j]) {
						temp = chisquares[j];
						chisquares[j] = chisquares[i];
						chisquares[i] = temp;
						tempindex = sorted_dim[j];
						sorted_dim[j] = sorted_dim[i];
						sorted_dim[i] = tempindex;
					}
				}
			}
			return sorted_dim;
		}


		public void setup(Context context) {
			mos = new MultipleOutputs(context);
			/** get configuration from file */
			Configuration conf = context.getConfiguration();
			num_dims = conf.getInt(SQConfig.strDimExpression, 2);
			cell_num = conf.getInt(SQConfig.strNumOfSmallCells, 501);
			domains = new float[2];
			domains[0] = conf.getFloat(SQConfig.strDomainMin, 0.0f);
			domains[1] = conf.getFloat(SQConfig.strDomainMax, 10001.0f);
			smallRange = (int) Math.ceil((domains[1] - domains[0]) / cell_num);
			di_numBuckets = conf.getInt(SQConfig.strNumOfPartitions, 2);
			partition_store = new float[(int) Math.pow(di_numBuckets, num_dims) + 1][num_dims * 2];
			partition_size = new int[(int) Math.pow(di_numBuckets, num_dims) + 1];
			start_end_points = new Hashtable<Double, StartAndEnd>();
		}

		/**
		 * default Reduce class.
		 * 
		 * @author Yizhou Yan
		 * @version Dec 31, 2015
		 */

		public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException {
			int num_data = 0; // number of data
			int[][] frequency = new int[num_dims][cell_num];
			Hashtable<String, Integer> mapData = new Hashtable<String, Integer>();

			if (Integer.parseInt(key.toString()) == 0) {
				// collect data
				for (Text oneValue : values) {
					String line = oneValue.toString();
					num_data++;
					String newLine = "";
					// rescale data and save to hashtable
					for (int i = 0; i < num_dims; i++) {
						float tempDataPerDim = Float.valueOf(line.split(",")[i + 1]);
						int indexDataPerDim = (int) (Math.floor(tempDataPerDim / smallRange));
						frequency[i][indexDataPerDim]++;
						newLine = newLine + indexDataPerDim * smallRange + ",";
					}
					newLine = newLine.substring(0, newLine.length() - 1);
					if (mapData.get(newLine) != null) {
						int tempFreq = mapData.get(newLine);
						mapData.put(newLine, tempFreq + 1);
					} else {
						mapData.put(newLine, 1);
					}
				} // end collection data
				double[] chisquares = new double[num_dims];

				// calculate chi-squares for each dimension
				for (int i = 0; i < num_dims; i++) {
					chisquares[i] = Cal_chisquare.Chi_square(frequency[i]);
				}

				// sort dimensions according to chi-squares
				int[] sorted_dim = sortDim(chisquares);

				// a queue to store partition plans
				Queue<PartitionPlan> queueOfPartition = new LinkedList<PartitionPlan>();

				// create object PartitionPlan and load all data into this plan
				float[] startOfRange = new float[num_dims];
				float[] endsOfRange = new float[num_dims];
				for (int i = 0; i < num_dims; i++) {
					startOfRange[i] = domains[0];
					endsOfRange[i] = domains[1];
				}

				PartitionPlan pp = new PartitionPlan();
				pp.setupPartitionPlan(num_dims, num_data, di_numBuckets, startOfRange, endsOfRange, cell_num,
						smallRange);
				pp.addDataPoints(mapData);
				// pp.printPartitionPlan();

				System.out.println("----------------------------Start 1st Partition---------------------------------");
				// Create equi-depth partitions for 1st dimension in dimList
				PartitionPlan[] newPP = new PartitionPlan[di_numBuckets];
				newPP = pp.seperatePartitions(sorted_dim[0], context.getConfiguration().getInt(SQConfig.strK, 3));
				int[] countExistPartitions = new int[num_dims];
				countExistPartitions[0] = 0;
				for (int i = 0; i < newPP.length; i++) {
					if (newPP[i].getNumData() != 0) {
						queueOfPartition.offer(newPP[i]);
						countExistPartitions[0]++;
						// newPP[i].printPartitionPlan();
					}
				}

				PartitionPlan[] PPs;
				// For each partition,

				for (int i = 1; i < num_dims; i++) {
					countExistPartitions[i] = 0;
					System.out.println(
							"------------------------------Start partition for " + (i + 1) + " th------------------");
					int newByDim = sorted_dim[i];

					for (int j = 0; j < countExistPartitions[i - 1]; j++) {
						System.out.println("Dealing with ...." + j);
						PartitionPlan needPartition = queueOfPartition.poll();
						// do not need to partition
						if (needPartition.getNumData() <= 2 * context.getConfiguration().getInt(SQConfig.strK, 3) + 1) { /////////////////////////////////
							// if (i == num_dims - 1) {
							String outputStr = needPartition.getStartAndEnd();
							if (outputStr.length() >= num_dims * 2) {
								String[] sub_splits = outputStr.split(",");
								// System.out.println(outputStr);
								for (int yy = 0; yy < num_dims * 2; yy++) {
									partition_store[indexForPartitionStore][yy] = Float.valueOf(sub_splits[yy]);
								}
								partition_size[indexForPartitionStore] = needPartition.getNumData();
								indexForPartitionStore++;
							}
						} else {
							PPs = new PartitionPlan[di_numBuckets];
							PPs = needPartition.seperatePartitions(newByDim,
									context.getConfiguration().getInt(SQConfig.strK, 3));

							for (int k = 0; k < PPs.length; k++) {
								if (i == num_dims - 1) {
									String outputStr = PPs[k].getStartAndEnd();
									// System.out.println(outputStr);
									if (outputStr.length() >= num_dims * 2) {
										String[] sub_splits = outputStr.split(",");
										for (int yy = 0; yy < num_dims * 2; yy++) {
											partition_store[indexForPartitionStore][yy] = Float
													.valueOf(sub_splits[yy]);
										}
										partition_size[indexForPartitionStore] = PPs[k].getNumData();
										indexForPartitionStore++;
									}
								} else {
									if (PPs[k].getNumData() != 0) {
										queueOfPartition.offer(PPs[k]);
										countExistPartitions[i]++;
										// PPs[k].printPartitionPlan();
									}

								}
							}
						}
					} // end for(int j = 0; j < multiplies; j++)
				}
			} // end if Integer.parseInt(key.toString()) == 0

			// sortBlocklist();
			// setupIndexes();

			String outputPPPath = context.getConfiguration().get(SQConfig.strPartitionPlanOutput);

			// String outputCellsPath =
			// context.getConfiguration().get(SQConfig.strCellsOutput);

			// output partition plan
			for (int i = 0; i < indexForPartitionStore; i++) {
				try {
					String str = "";
					str += i + ",";
					for (int j = 0; j < num_dims * 2; j++) {
						str += partition_store[i][j] + ",";
					}
					str = str.substring(0, str.length() - 1);
					mos.write(NullWritable.get(), new Text(str), outputPPPath + "/pp");
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		} // end reduce function

		public void cleanup(Context context) throws IOException, InterruptedException {
			mos.close();
		}
	} // end reduce class

	public void run(String[] args) throws Exception {
		Configuration conf = new Configuration();
		conf.addResource(new Path("/usr/local/Cellar/hadoop/etc/hadoop/core-site.xml"));
		conf.addResource(new Path("/usr/local/Cellar/hadoop/etc/hadoop/hdfs-site.xml"));
		new GenericOptionsParser(conf, args).getRemainingArgs();
		/** set job parameter */
		Job job = Job.getInstance(conf, "Distributed Data Driven Sampling");

		job.setJarByClass(DataDrivenPartition.class);
		job.setMapperClass(DDMapper.class);

		/** set multiple output path */
		MultipleOutputs.addNamedOutput(job, "partitionplan", TextOutputFormat.class, NullWritable.class, Text.class);

		job.setReducerClass(DDReducer.class);
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(Text.class);
		job.setOutputKeyClass(NullWritable.class);
		job.setOutputValueClass(Text.class);
		job.setNumReduceTasks(1);

		String strFSName = conf.get("fs.default.name");
		FileInputFormat.addInputPath(job, new Path(conf.get(SQConfig.dataset)));
		FileSystem fs = FileSystem.get(conf);
		fs.delete(new Path(conf.get(SQConfig.strSamplingOutput)), true);
		FileOutputFormat.setOutputPath(job, new Path(conf.get(SQConfig.strSamplingOutput)));

		/** print job parameter */
		System.err.println("# of dim: " + conf.getInt(SQConfig.strDimExpression, 10));
		long begin = System.currentTimeMillis();
		job.waitForCompletion(true);
		long end = System.currentTimeMillis();
		long second = (end - begin) / 1000;
		System.err.println(job.getJobName() + " takes " + second + " seconds");
	}

	public static void main(String[] args) throws Exception {
		DataDrivenPartition DDPartition = new DataDrivenPartition();
		DDPartition.run(args);
	}
}
