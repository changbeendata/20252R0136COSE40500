"""
Main Pipeline for TELEClass: Taxonomy Enrichment and LLM-Enhanced Hierarchical Text Classification
Orchestrates the full three-phase pipeline.
"""

import os
import argparse
import json
import random
import numpy as np
import torch
from datetime import datetime

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from taxonomy import Taxonomy
from enrichment import enrich_taxonomy_full
from annotation import run_annotation_pipeline, TaxonomyEmbedder
from train import train_classifier, predict, HierarchicalTextClassifier
from transformers import AutoTokenizer


def phase1_enrichment(args):
    """Phase 1: Taxonomy Enrichment."""
    print("\n" + "=" * 80)
    print(" PHASE 1: TAXONOMY ENRICHMENT ".center(80, "="))
    print("=" * 80 + "\n")
    
    # Load base taxonomy
    print("Loading base taxonomy...")
    taxonomy = Taxonomy()
    taxonomy.load_from_files(
        classes_path=args.classes_path,
        hierarchy_path=args.hierarchy_path,
        keywords_path=args.keywords_path
    )
    print(f"✓ Loaded taxonomy: {taxonomy}")
    print(f"  Statistics: {taxonomy.get_statistics()}")
    
    # Enrich taxonomy
    enriched_taxonomy = enrich_taxonomy_full(
        taxonomy=taxonomy,
        corpus_path=args.train_corpus_path,
        api_key=args.openai_api_key,
        llm_cache_path=args.llm_cache_path,
        output_path=args.enriched_taxonomy_path,
        max_llm_calls=args.max_llm_calls,
        num_workers=args.num_llm_workers,
        use_llm_for_enrichment=args.use_llm_for_enrichment
    )
    
    print(f"\n✓ Phase 1 complete! Enriched taxonomy saved to: {args.enriched_taxonomy_path}")
    return enriched_taxonomy


def phase2_annotation(args):
    """Phase 2: Pseudo-Label Generation."""
    print("\n" + "=" * 80)
    print(" PHASE 2: PSEUDO-LABEL GENERATION ".center(80, "="))
    print("=" * 80 + "\n")
    
    # Run annotation pipeline
    pseudo_labels = run_annotation_pipeline(
        enriched_taxonomy_path=args.enriched_taxonomy_path,
        corpus_path=args.train_corpus_path,
        use_llm_verification=args.use_llm_verification,
        api_key=args.openai_api_key,
        output_path=args.pseudo_labels_path,
        embedder_model=args.embedder_model,
        max_llm_calls=args.max_llm_calls,
        llm_sample_rate=args.llm_sample_rate,
        num_workers=args.num_llm_workers,
        device=args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    print(f"\n✓ Phase 2 complete! Pseudo-labels saved to: {args.pseudo_labels_path}")
    return pseudo_labels


def phase3_training(args):
    """Phase 3: Classifier Training."""
    print("\n" + "=" * 80)
    print(" PHASE 3: CLASSIFIER TRAINING ".center(80, "="))
    print("=" * 80 + "\n")
    
    # Train classifier
    model = train_classifier(
        train_corpus_path=args.train_corpus_path,
        train_labels_path=args.pseudo_labels_path,
        val_corpus_path=args.val_corpus_path if args.val_corpus_path else None,
        val_labels_path=args.val_labels_path if args.val_labels_path else None,
        model_name=args.classifier_model,
        num_classes=args.num_classes,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        output_dir=args.model_output_dir,
        device=args.device,
        use_hierarchical_loss=args.use_hierarchical_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        hierarchy_weight=args.hierarchy_weight,
        root_weight=args.root_weight,
        taxonomy_path=args.enriched_taxonomy_path
    )
    
    print(f"\n✓ Phase 3 complete! Model saved to: {args.model_output_dir}")
    return model


def evaluate_predictions(args):
    """Evaluate predictions against ground truth."""
    print("\n" + "=" * 80)
    print(" EVALUATING PREDICTIONS ".center(80, "="))
    print("=" * 80 + "\n")
    
    from evaluate_submission import evaluate
    
    evaluate(
        ground_truth_path=args.ground_truth_path,
        predictions_path=args.test_predictions_path,
        num_classes=args.num_classes
    )


def generate_test_predictions(args):
    """Generate predictions on test set."""
    print("\n" + "=" * 80)
    print(" GENERATING TEST PREDICTIONS ".center(80, "="))
    print("=" * 80 + "\n")
    
    # Setup device
    device = torch.device(args.device if args.device else 
                         ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading trained model from {args.model_output_dir}...")
    model = HierarchicalTextClassifier(
        model_name=args.classifier_model,
        num_classes=args.num_classes
    )
    
    model_path = os.path.join(args.model_output_dir, "best_model.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(args.model_output_dir, "final_model.pt")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_output_dir)
    
    # Load test corpus
    print(f"\nLoading test corpus from {args.test_corpus_path}...")
    test_corpus = {}
    with open(args.test_corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                doc_id, text = parts
                test_corpus[doc_id] = text
    
    print(f"Loaded {len(test_corpus)} test documents")
    
    # Make predictions
    print("\nGenerating predictions...")
    doc_ids = list(test_corpus.keys())
    texts = [test_corpus[doc_id] for doc_id in doc_ids]
    
    predictions = predict(
        model=model,
        texts=texts,
        tokenizer=tokenizer,
        device=device,
        max_length=args.max_length,
        top_k=args.prediction_top_k,
        threshold=args.prediction_threshold
    )
    
    # Save predictions
    print(f"\nSaving predictions to {args.test_predictions_path}...")
    
    # Save as CSV for Kaggle submission
    import csv
    with open(args.test_predictions_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['pid', 'labels'])
        for doc_id, pred_labels in zip(doc_ids, predictions):
            writer.writerow([doc_id, ','.join(map(str, pred_labels))])
    
    print(f"✓ Test predictions saved!")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Avg labels per doc: {sum(len(p) for p in predictions) / len(predictions):.2f}")
    
    return predictions


def run_full_pipeline(args):
    """Run the complete TELEClass pipeline."""
    print("\n" + "=" * 80)
    print(" TELEClass FULL PIPELINE ".center(80, "="))
    print("=" * 80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Phase 1: Taxonomy Enrichment
    if not args.skip_enrichment:
        phase1_enrichment(args)
    else:
        print("\n⊳ Skipping Phase 1 (using existing enriched taxonomy)")
    
    # Phase 2: Pseudo-Label Generation
    if not args.skip_annotation:
        phase2_annotation(args)
    else:
        print("\n⊳ Skipping Phase 2 (using existing pseudo-labels)")
    
    # Phase 3: Classifier Training
    if not args.skip_training:
        phase3_training(args)
    else:
        print("\n⊳ Skipping Phase 3 (using existing model)")
    
    # Generate test predictions
    if args.generate_test_predictions:
        generate_test_predictions(args)
    
    # Evaluate predictions if ground truth available
    if args.evaluate_predictions and os.path.exists(args.ground_truth_path):
        evaluate_predictions(args)
    
    print("\n" + "=" * 80)
    print(" PIPELINE COMPLETE ".center(80, "="))
    print("=" * 80)
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    parser = argparse.ArgumentParser(
        description="TELEClass: Taxonomy Enrichment and LLM-Enhanced Hierarchical Text Classification"
    )
    
    # === General Arguments ===
    parser.add_argument("--mode", type=str, default="full",
                       choices=["full", "phase1", "phase2", "phase3", "predict", "evaluate"],
                       help="Pipeline mode to run")
    
    # === Data Paths ===
    parser.add_argument("--classes_path", type=str, 
                       default="Amazon_products/classes.txt",
                       help="Path to classes file")
    parser.add_argument("--hierarchy_path", type=str,
                       default="Amazon_products/class_hierarchy.txt",
                       help="Path to hierarchy file")
    parser.add_argument("--keywords_path", type=str,
                       default="Amazon_products/class_related_keywords.txt",
                       help="Path to keywords file")
    parser.add_argument("--train_corpus_path", type=str,
                       default="Amazon_products/train/train_corpus.txt",
                       help="Path to training corpus")
    parser.add_argument("--test_corpus_path", type=str,
                       default="Amazon_products/test/test_corpus.txt",
                       help="Path to test corpus")
    parser.add_argument("--val_corpus_path", type=str, default=None,
                       help="Path to validation corpus (optional)")
    parser.add_argument("--val_labels_path", type=str, default=None,
                       help="Path to validation labels (optional)")
    
    # === Output Paths ===
    parser.add_argument("--enriched_taxonomy_path", type=str,
                       default="outputs/taxonomy_enriched.json",
                       help="Path to save/load enriched taxonomy")
    parser.add_argument("--pseudo_labels_path", type=str,
                       default="outputs/pseudo_labels_train.json",
                       help="Path to save/load pseudo-labels")
    parser.add_argument("--model_output_dir", type=str,
                       default="outputs/models/teleclass",
                       help="Directory to save trained model")
    parser.add_argument("--test_predictions_path", type=str,
                       default="outputs/submission.csv",
                       help="Path to save test predictions")
    parser.add_argument("--ground_truth_path", type=str,
                       default="doc2labels.txt",
                       help="Path to ground truth labels for evaluation")
    parser.add_argument("--evaluate_predictions", action="store_true",
                       help="Evaluate predictions against ground truth after generation")
    
    # === Phase 1: Enrichment Arguments ===
    parser.add_argument("--openai_api_key", type=str, default=None,
                       help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--llm_cache_path", type=str,
                       default="outputs/llm_enrichment_cache.json",
                       help="Path to LLM cache")
    parser.add_argument("--max_llm_calls", type=int, default=1000,
                       help="Maximum number of LLM API calls allowed (default: 1000)")
    parser.add_argument("--num_llm_workers", type=int, default=8,
                       help="Number of parallel workers for LLM calls (default: 8)")
    parser.add_argument("--use_llm_for_enrichment", action="store_true",
                       help="Use LLM for Phase 1 enrichment (default: False, uses keywords file)")
    
    # === Phase 2: Annotation Arguments ===
    parser.add_argument("--embedder_model", type=str,
                       default="all-MiniLM-L6-v2",
                       help="Sentence-transformers model for embeddings")
    parser.add_argument("--use_llm_verification", action="store_true",
                       help="Use LLM for candidate verification in Phase 2")
    parser.add_argument("--llm_sample_rate", type=float, default=0.05,
                       help="Rate at which to sample documents for LLM verification (0-1, default: 0.05)")
    
    # === Phase 3: Training Arguments ===
    parser.add_argument("--classifier_model", type=str,
                       default="distilbert-base-uncased",
                       help="Hugging Face model for classifier")
    parser.add_argument("--num_classes", type=int, default=531,
                       help="Number of classes in taxonomy")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu)")
    
    # Loss function arguments
    parser.add_argument("--use_hierarchical_loss", action="store_true", default=True,
                       help="Use hierarchical loss with focal loss (default: True)")
    parser.add_argument("--focal_alpha", type=float, default=0.25,
                       help="Focal loss alpha parameter (default: 0.25)")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                       help="Focal loss gamma parameter (default: 2.0)")
    parser.add_argument("--hierarchy_weight", type=float, default=0.3,
                       help="Weight for hierarchical consistency loss (default: 0.3)")
    parser.add_argument("--root_weight", type=float, default=3.0,
                       help="Extra weight for root nodes (default: 3.0)")
    
    # === Prediction Arguments ===
    parser.add_argument("--prediction_top_k", type=int, default=3,
                       help="Maximum number of labels per prediction")
    parser.add_argument("--prediction_threshold", type=float, default=0.1,
                       help="Probability threshold for predictions (default: 0.3, lower = more labels)")
    parser.add_argument("--use_hierarchical_masking", action="store_true", default=True,
                       help="Use hierarchical top-down masking for prediction (default: True)")
    parser.add_argument("--no_hierarchical_masking", action="store_true",
                       help="Disable hierarchical masking (use flat prediction)")
    
    # === Pipeline Control ===
    parser.add_argument("--skip_enrichment", action="store_true",
                       help="Skip Phase 1 (use existing enriched taxonomy)")
    parser.add_argument("--skip_annotation", action="store_true",
                       help="Skip Phase 2 (use existing pseudo-labels)")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip Phase 3 (use existing model)")
    parser.add_argument("--generate_test_predictions", action="store_true",
                       help="Generate predictions on test set after training")
    
    args = parser.parse_args()
    
    # Set OpenAI API key from env if not provided
    if args.openai_api_key is None:
        args.openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Create output directories
    enriched_dir = os.path.dirname(args.enriched_taxonomy_path)
    if enriched_dir:
        os.makedirs(enriched_dir, exist_ok=True)
    
    pseudo_labels_dir = os.path.dirname(args.pseudo_labels_path)
    if pseudo_labels_dir:
        os.makedirs(pseudo_labels_dir, exist_ok=True)
    
    os.makedirs(args.model_output_dir, exist_ok=True)
    
    test_pred_dir = os.path.dirname(args.test_predictions_path)
    if test_pred_dir:
        os.makedirs(test_pred_dir, exist_ok=True)
    
    # Run appropriate mode
    if args.mode == "full":
        run_full_pipeline(args)
    elif args.mode == "phase1":
        phase1_enrichment(args)
    elif args.mode == "phase2":
        phase2_annotation(args)
    elif args.mode == "phase3":
        phase3_training(args)
    elif args.mode == "predict":
        generate_test_predictions(args)
        if args.evaluate_predictions and os.path.exists(args.ground_truth_path):
            evaluate_predictions(args)
    elif args.mode == "evaluate":
        evaluate_predictions(args)


if __name__ == "__main__":
    main()
