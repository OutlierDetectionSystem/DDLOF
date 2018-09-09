package metricspace;

import java.util.HashMap;
import java.util.Map;

@SuppressWarnings("rawtypes")
public class MetricObject implements Comparable {
	private int partition_id;
	private char type;
	// for second or more use
	private char orgType;
	private float distToPivot;
	private Object obj;
	private Map<Long,Float> knnInDetail = new HashMap<Long,Float>();
	private float kdist = 0;
	private float expandDist = 0.0f; 
	private String knnsInString = "";
	private float nearestNeighborDist = Float.MAX_VALUE;
	private boolean canPrune = false;
	private float lrd = 0;
	private float lof = 0;
	
	public boolean isCanPrune() {
		return canPrune;
	}

	public void setCanPrune(boolean canPrune) {
		this.canPrune = canPrune;
	}

	public float getNearestNeighborDist() {
		return nearestNeighborDist;
	}

	public void setNearestNeighborDist(float nearestNeighborDist) {
		this.nearestNeighborDist = nearestNeighborDist;
	}
	
	
	public float getLrd() {
		return lrd;
	}

	public void setLrd(float lrd) {
		this.lrd = lrd;
	}

	public float getLof() {
		return lof;
	}

	public void setLof(float lof) {
		this.lof = lof;
	}
	private String whoseSupport="";
	private boolean canCalculateLof = false;
	public MetricObject() {
	}

	public MetricObject (int partition_id, Object obj){
		this.partition_id = partition_id;
		this.obj = obj;
	}
	
	public MetricObject (int partition_id, Object obj, char type){
		this(partition_id, obj);
		this.type = type;
	}
	
	public MetricObject (int partition_id, Object obj, char type, char orgType, float kdist, float lrd, float lof){
		this(partition_id, obj);
		this.type = type;
		this.orgType = orgType;
		this.kdist = kdist;
		this.lrd = lrd;
		this.lof = lof;
	}
	public MetricObject(int partition_id, Object obj, float curKdist, Map<Long, Float> knnInDetail, 
			char curTag, String whoseSupport){
		this(partition_id,obj, curTag);
		this.kdist = curKdist;
		this.knnInDetail = knnInDetail;
		this.whoseSupport = whoseSupport;
	}
	public MetricObject(int partition_id, Object obj, float curKdist, String knns, char curTag, String whoseSupport){
		this(partition_id, obj, curTag);
		this.kdist = curKdist;
		this.knnsInString =  knns;
		this.whoseSupport = whoseSupport;
	}
	public String toString() {
		StringBuilder sb = new StringBuilder(); 
		Record r = (Record) obj;
		
		sb.append(",type: "+type);
		sb.append(",in Partition: "+partition_id);
		
		sb.append(", Knn in detail: ");
		
		for (Long v : knnInDetail.keySet()) {
			sb.append("," + v + "," + knnInDetail.get(v));
		}
		return sb.toString();
	}
	
	/**
	 * sort by the descending order
	 */
	@Override
	public int compareTo(Object o) {
		MetricObject other = (MetricObject) o;
		if (other.distToPivot > this.distToPivot)
			return 1;
		else if (other.distToPivot < this.distToPivot)
			return -1;
		else
			return 0;
	}

	public int getPartition_id() {
		return partition_id;
	}

	public void setPartition_id(int partition_id) {
		this.partition_id = partition_id;
	}

	public char getType() {
		return type;
	}

	public void setType(char type) {
		this.type = type;
	}

	public float getDistToPivot() {
		return distToPivot;
	}

	public void setDistToPivot(float distToPivot) {
		this.distToPivot = distToPivot;
	}

	public Object getObj() {
		return obj;
	}

	public void setObj(Object obj) {
		this.obj = obj;
	}

	public String getWhoseSupport() {
		return whoseSupport;
	}

	public void setWhoseSupport(String whoseSupport) {
		this.whoseSupport = whoseSupport;
	}

	public Map<Long, Float> getKnnInDetail() {
		return knnInDetail;
	}

	public void setKnnInDetail(Map<Long, Float> knnInDetail) {
		this.knnInDetail = knnInDetail;
	}

	public float getKdist() {
		return kdist;
	}

	public void setKdist(float kdist) {
		this.kdist = kdist;
	}


	public boolean isCanCalculateLof() {
		return canCalculateLof;
	}

	public void setCanCalculateLof(boolean canCalculateLof) {
		this.canCalculateLof = canCalculateLof;
	}
	
	public float getExpandDist() {
		return expandDist;
	}

	public void setExpandDist(float expandDist) {
		this.expandDist = expandDist;
	}
	public String getKnnsInString() {
		return knnsInString;
	}

	public void setKnnsInString(String knnsInString) {
		this.knnsInString = knnsInString;
	}
	
}
