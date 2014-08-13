package edu.csupomona.nlp.ml.supervised;

public class NaiveBayesResult {
	private String label;
	private double probability;
	
	public NaiveBayesResult(String label, double probability) {
		super();
		this.label = label;
		this.probability = probability;
	}
	public String getLabel() {
		return label;
	}
	public void setLabel(String label) {
		this.label = label;
	}
	public double getProbability() {
		return probability;
	}
	public void setProbability(double probability) {
		this.probability = probability;
	}
		
}
