package com.hadoop;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordWriter;
import org.apache.hadoop.mapred.lib.MultipleOutputFormat;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Progressable;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import  java.util.List;
public class kMeans {
    public static class MapkMeans extends Mapper<Object, Text, IntWritable, Text>{
        private IntWritable ClusterID = new IntWritable(); //初始化簇ID
        private final static IntWritable one = new IntWritable(1);
        private Text pointPair = new Text(); //初始化(p#1)
        private List<double[]> Centers = new ArrayList<>(); // Centers[i] = 向量的16个分量
        // 计算两点间的欧式距离
        private double DistCount(double[] center, String[] point){//计算欧式距离
            double[] tempPoint = new double[point.length];
            double sum = 0;
            for(int i = 0; i < point.length; i++){
                tempPoint[i] = Double.parseDouble(point[i]);
            }
            if(center.length != tempPoint.length){
                System.out.println("different length");
            }
            else{
                int len = center.length;
                for(int i = 0; i < len; i++){
                    sum += Math.pow((center[i] - tempPoint[i]), 2);
                }
                sum = Math.sqrt(sum);
            }
            return sum;
        }

        @Override
        protected void setup(Context context) throws IOException, InterruptedException{
            //读取全局聚类中心数据？
            Configuration conf = context.getConfiguration();
            String currCenterPath = conf.get("centerPath"); //获取聚类中心信息文件
            BufferedReader BufRead = new BufferedReader(new FileReader(currCenterPath)); //使用行缓冲读取，按行处理文件
            String line = new String();
            while((line = BufRead.readLine()) != null){
                String[] CenterStr = line.split(":")[1].trim().split(","); //取文件中向量的分量部分
                double[] CenterDouble = new double[CenterStr.length];
                for(int i = 0; i < CenterStr.length; i++){
                    CenterDouble[i] = Double.parseDouble(CenterStr[i].trim());
                } // 使用trim方法去除空格后，转为double，放入centers中作为一个向量
                Centers.add(CenterDouble);
            }
            BufRead.close();
        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            double disMin = Double.MAX_VALUE;
            int idx = -1;
            // 把每个point按逗号切割为若干分量
            String[] point = value.toString().split(":")[1].trim().split(",");
            // 找到当前点可能属于的聚类中心,也即计算最小距离
            for(int i = 0; i < Centers.size(); i++){
                double tempDis = DistCount(Centers.get(i), point);
                if(tempDis < disMin){
                    disMin = tempDis;
                    idx = i;
                }
            }
            ClusterID.set(idx); // Center[i].clusterID
            pointPair.set(value.toString().split(":")[1].trim() + "#" + one.toString()); //构造(p#1)的pair
            context.write(ClusterID, pointPair);
        }
//选做部分，使用multiple output format进行输出文件自定义
        public static class SaveByClusterOutputFormat extends MultipleOutputFormat<IntWritable, Text>{
            protected String generateFileNameForKeyValue(IntWritable key, Text value, String filename){
                String cluster = key.toString();
                return cluster + "/" + filename;
            }
            @Override
            protected RecordWriter<IntWritable, Text> getBaseRecordWriter(FileSystem fileSystem, JobConf jobConf, String s, Progressable progressable) throws IOException {
                return null;
            }
        }
    }

    public class CombinekMeans extends Reducer<IntWritable, Text, IntWritable, Text>{
        private final static int len = 16; // 分量个数：16
        private Text pointPair = new Text();
        public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            double[] pm = new double[len];
            int num = 0;
            for(Text val: values){//针对(k2,[v2])的[v2]中的每个v2进行操作
                //前文有提到，我们的value pair格式为(p#1)，其中p = x1,x2,x3...
                //两次分割，取出数据点的分量部分
                String[] ValList = val.toString().split("#")[0].trim().split(",");
                for(int i = 0; i < len; i++){
                    pm[i] += Double.parseDouble(ValList[i]);//对应分量累加到pm之中
                }
                num += Integer.parseInt(val.toString().split("#")[1].trim());//数据点的个数部分累加到num中
            }
            for(int i = 0; i < pm.length; i++){
                pm[i] = pm[i] / num;//求均值 pm / n
            }
            StringBuilder ValString = new StringBuilder(new String());
            for(int i = 0; i < len; i++){
                ValString.append(",").append(pm[i]);
            }
            ValString.append("#").append(num);//把#补回去
            pointPair.set(ValString.toString());
            context.write(key, pointPair);
        }
    }

    public class ReducekMeans extends Reducer<IntWritable, Text, IntWritable, Text>{
        private final static int len = 16;
        private Text pointPair = new Text();
        //原理和combiner类似，只是在算均值时计算方式略有不同
        public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException{
            double[] pm = new double[len];
            int num = 0;
            for(Text val: values){
                String[] ValList = val.toString().split("#")[0].trim().split(",");
                int bonus = Integer.parseInt(val.toString().split("#")[1].trim());//提取个数
                for(int i = 0; i < len; i++){
                    pm[i] = pm[i] + Double.parseDouble(ValList[i]) * bonus;//需要乘上各分量为当前值的数据点个数
                }
                num += bonus;//需要加上各分量为当前值的数据点个数
            }
            for(int i = 0; i < pm.length; i++){
                pm[i] = pm[i] / num;
            }
            StringBuilder ValString = new StringBuilder(new String());
            for(int i = 0; i < len; i++){
                ValString.append(",").append(pm[i]);
            }
            ValString.append("#").append(num);
            pointPair.set(ValString.toString());
            context.write(key, pointPair);
        }
    }
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        conf.set("centerPath", args[2]);//设置中心读取路径
        Job job = Job.getInstance(conf, "K-MEANS");
        job.setJarByClass(kMeans.class);
        job.setMapperClass(MapkMeans.class);
        job.setCombinerClass(CombinekMeans.class);
        job.setReducerClass(ReducekMeans.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
