import os
import shutil
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
import torch
import tensorflow as tf
import numpy as np

def convert_safetensors_to_tflite(
    model_dir="./model",
    tflite_output_path="./model.tflite",
    max_length=128
):
    """
    Convertit un modèle SafeTensors en TFLite en passant par TensorFlow directement
    
    Args:
        model_dir: Dossier contenant model.safetensors + config.json
        tflite_output_path: Chemin de sortie pour le modèle TFLite
        max_length: Longueur maximale des séquences
    """
    
    try:
        # === 1. Nettoyage des dossiers précédents ===
        tf_model_dir = "./tf_model"
        if os.path.exists(tf_model_dir):
            shutil.rmtree(tf_model_dir)
            print(f"🗑️ Dossier {tf_model_dir} supprimé")
        
        # === 2. Charger le modèle PyTorch ===
        print("📥 Chargement du modèle PyTorch depuis les fichiers locaux (.safetensors)...")
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        pytorch_model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, 
            config=config, 
            local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        
        print(f"✅ Modèle PyTorch chargé : {config.model_type}")
        print(f"📊 Nombre de labels : {config.num_labels}")
        
        # === 3. Créer le modèle TensorFlow équivalent ===
        print("🔄 Création du modèle TensorFlow...")
        
        # Créer le modèle TF avec la même configuration
        tf_model = TFAutoModelForSequenceClassification.from_config(config)
        
        # === 4. Transférer les poids de PyTorch vers TensorFlow ===
        print("🔄 Transfert des poids PyTorch vers TensorFlow...")
        
        # Préparer des données factices pour initialiser le modèle TF
        dummy_inputs = {
            'input_ids': tf.constant([[1, 2, 3, 4, 5]], dtype=tf.int32),
            'attention_mask': tf.constant([[1, 1, 1, 1, 1]], dtype=tf.int32)
        }
        
        # Faire un passage avant pour initialiser les couches
        _ = tf_model(dummy_inputs)
        
        # Transférer les poids manuellement
        pytorch_state_dict = pytorch_model.state_dict()
        
        # Créer un mapping des poids PyTorch vers TensorFlow
        weight_mapping = create_weight_mapping(pytorch_model, tf_model)
        
        for tf_var, pt_param_name in weight_mapping.items():
            if pt_param_name in pytorch_state_dict:
                pt_weight = pytorch_state_dict[pt_param_name].detach().numpy()
                
                # Ajuster la forme si nécessaire (pour les couches linéaires)
                if len(pt_weight.shape) == 2 and len(tf_var.shape) == 2:
                    if pt_weight.shape != tf_var.shape:
                        pt_weight = pt_weight.T
                
                try:
                    tf_var.assign(pt_weight)
                    print(f"✅ Poids transférés : {pt_param_name} -> {tf_var.name}")
                except Exception as e:
                    print(f"⚠️ Erreur transfert {pt_param_name}: {e}")
        
        # === 5. Créer un modèle avec signature concrète ===
        print("🔧 Création du modèle avec signature concrète...")
        
        @tf.function
        def serving_fn(input_ids, attention_mask):
            outputs = tf_model({
                'input_ids': input_ids,
                'attention_mask': attention_mask
            })
            return outputs.logits
        
        # Définir la signature concrète
        concrete_function = serving_fn.get_concrete_function(
            input_ids=tf.TensorSpec(shape=[None, max_length], dtype=tf.int32, name='input_ids'),
            attention_mask=tf.TensorSpec(shape=[None, max_length], dtype=tf.int32, name='attention_mask')
        )
        
        # === 6. Sauvegarder le modèle TensorFlow ===
        print("💾 Sauvegarde du modèle TensorFlow...")
        tf.saved_model.save(
            tf_model,
            tf_model_dir,
            signatures={'serving_default': concrete_function}
        )
        print(f"✅ Modèle TensorFlow sauvegardé : {tf_model_dir}")
        
        # === 7. Conversion vers TFLite ===
        print("🔄 Conversion vers TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
        
        # Configuration du convertisseur
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS  # Permet d'utiliser certaines ops TF non supportées nativement
        ]
        
        # Fonction représentative pour la quantification (optionnelle)
        def representative_dataset():
            for _ in range(100):
                # Créer des données aléatoires représentatives
                input_ids = np.random.randint(1, 1000, size=(1, max_length), dtype=np.int32)
                attention_mask = np.ones((1, max_length), dtype=np.int32)
                yield [input_ids, attention_mask]
        
        # Essayer la conversion avec optimisations
        try:
            converter.representative_dataset = representative_dataset
            tflite_model = converter.convert()
            print("✅ Conversion avec optimisations réussie")
        except Exception as e:
            print(f"⚠️ Échec avec optimisations: {e}")
            print("🔄 Tentative sans optimisations...")
            
            # Nouvelle tentative sans optimisations
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            tflite_model = converter.convert()
            print("✅ Conversion sans optimisations réussie")
        
        # === 8. Sauvegarde du modèle TFLite ===
        with open(tflite_output_path, "wb") as f:
            f.write(tflite_model)
        
        # Informations sur le modèle final
        model_size = os.path.getsize(tflite_output_path) / (1024 * 1024)  # MB
        print(f"✅ Conversion terminée !")
        print(f"📁 Modèle TFLite : {tflite_output_path}")
        print(f"📏 Taille du modèle : {model_size:.2f} MB")
        
        # === 9. Test du modèle TFLite ===
        print("🧪 Test du modèle TFLite...")
        test_tflite_model(tflite_output_path, tokenizer, max_length)
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la conversion : {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_weight_mapping(pytorch_model, tf_model):
    """
    Crée un mapping entre les poids PyTorch et TensorFlow
    """
    weight_mapping = {}
    
    # Obtenir tous les paramètres nommés
    pt_params = dict(pytorch_model.named_parameters())
    tf_vars = {var.name: var for var in tf_model.trainable_variables}
    
    # Mapping de base pour BERT
    base_mappings = {
        'embeddings.word_embeddings.weight': 'embeddings/word_embeddings/weight:0',
        'embeddings.position_embeddings.weight': 'embeddings/position_embeddings/weight:0',
        'embeddings.token_type_embeddings.weight': 'embeddings/token_type_embeddings/weight:0',
        'embeddings.LayerNorm.weight': 'embeddings/layer_norm/gamma:0',
        'embeddings.LayerNorm.bias': 'embeddings/layer_norm/beta:0',
        'classifier.weight': 'classifier/weight:0',
        'classifier.bias': 'classifier/bias:0',
    }
    
    # Ajouter les mappings pour les couches d'attention et feed-forward
    for i in range(12):  # BERT-base a généralement 12 couches
        layer_mappings = {
            f'encoder.layer.{i}.attention.self.query.weight': f'encoder/layer_._{i}/attention/self/query/weight:0',
            f'encoder.layer.{i}.attention.self.query.bias': f'encoder/layer_._{i}/attention/self/query/bias:0',
            f'encoder.layer.{i}.attention.self.key.weight': f'encoder/layer_._{i}/attention/self/key/weight:0',
            f'encoder.layer.{i}.attention.self.key.bias': f'encoder/layer_._{i}/attention/self/key/bias:0',
            f'encoder.layer.{i}.attention.self.value.weight': f'encoder/layer_._{i}/attention/self/value/weight:0',
            f'encoder.layer.{i}.attention.self.value.bias': f'encoder/layer_._{i}/attention/self/value/bias:0',
            f'encoder.layer.{i}.attention.output.dense.weight': f'encoder/layer_._{i}/attention/output/dense/weight:0',
            f'encoder.layer.{i}.attention.output.dense.bias': f'encoder/layer_._{i}/attention/output/dense/bias:0',
            f'encoder.layer.{i}.attention.output.LayerNorm.weight': f'encoder/layer_._{i}/attention/output/layer_norm/gamma:0',
            f'encoder.layer.{i}.attention.output.LayerNorm.bias': f'encoder/layer_._{i}/attention/output/layer_norm/beta:0',
            f'encoder.layer.{i}.intermediate.dense.weight': f'encoder/layer_._{i}/intermediate/dense/weight:0',
            f'encoder.layer.{i}.intermediate.dense.bias': f'encoder/layer_._{i}/intermediate/dense/bias:0',
            f'encoder.layer.{i}.output.dense.weight': f'encoder/layer_._{i}/output/dense/weight:0',
            f'encoder.layer.{i}.output.dense.bias': f'encoder/layer_._{i}/output/dense/bias:0',
            f'encoder.layer.{i}.output.LayerNorm.weight': f'encoder/layer_._{i}/output/layer_norm/gamma:0',
            f'encoder.layer.{i}.output.LayerNorm.bias': f'encoder/layer_._{i}/output/layer_norm/beta:0',
        }
        base_mappings.update(layer_mappings)
    
    # Créer le mapping final
    for pt_name, tf_name in base_mappings.items():
        if pt_name in pt_params and tf_name in tf_vars:
            weight_mapping[tf_vars[tf_name]] = pt_name
    
    return weight_mapping

def test_tflite_model(tflite_path, tokenizer, max_length):
    """
    Test rapide du modèle TFLite
    """
    try:
        # Charger le modèle TFLite
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Obtenir les détails des entrées et sorties
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"📋 Entrées du modèle TFLite :")
        for i, detail in enumerate(input_details):
            print(f"  {i}: {detail['name']} - Shape: {detail['shape']} - Type: {detail['dtype']}")
        
        print(f"📋 Sorties du modèle TFLite :")
        for i, detail in enumerate(output_details):
            print(f"  {i}: {detail['name']} - Shape: {detail['shape']} - Type: {detail['dtype']}")
        
        # Préparer les données de test
        test_text = "This is a test sentence for model conversion."
        inputs = tokenizer(
            test_text,
            return_tensors="np",
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
        
        # Faire une prédiction
        interpreter.set_tensor(input_details[0]['index'], inputs['input_ids'].astype(np.int32))
        interpreter.set_tensor(input_details[1]['index'], inputs['attention_mask'].astype(np.int32))
        
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output_data[0])
        
        print(f"✅ Test réussi ! Forme de sortie : {output_data.shape}")
        print(f"📊 Classe prédite : {predicted_class}")
        print(f"📊 Scores (5 premiers) : {output_data[0][:5]}")
        
    except Exception as e:
        print(f"⚠️ Erreur lors du test : {str(e)}")

def convert_with_alternative_method(model_dir, tflite_output_path, max_length=128):
    """
    Méthode alternative utilisant tf.keras directement
    """
    print("🔄 Tentative avec méthode alternative (tf.keras)...")
    
    try:
        # Charger le modèle avec from_pretrained directement en TensorFlow
        print("📥 Chargement direct du modèle TensorFlow...")
        tf_model = TFAutoModelForSequenceClassification.from_pretrained(
            model_dir, 
            from_tf=False,  # Convertir depuis PyTorch
            local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        
        # Créer un modèle Keras fonctionnel
        input_ids = tf.keras.Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = tf.keras.Input(shape=(max_length,), dtype=tf.int32, name='attention_mask')
        
        outputs = tf_model({'input_ids': input_ids, 'attention_mask': attention_mask})
        keras_model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=outputs.logits)
        
        # Conversion directe vers TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        tflite_model = converter.convert()
        
        with open(tflite_output_path, "wb") as f:
            f.write(tflite_model)
        
        model_size = os.path.getsize(tflite_output_path) / (1024 * 1024)
        print(f"✅ Conversion alternative réussie !")
        print(f"📁 Modèle TFLite : {tflite_output_path}")
        print(f"📏 Taille du modèle : {model_size:.2f} MB")
        
        # Test du modèle
        test_tflite_model(tflite_output_path, tokenizer, max_length)
        
        return True
        
    except Exception as e:
        print(f"❌ Méthode alternative échouée : {str(e)}")
        return False

if __name__ == "__main__":
    # === Configuration ===
    MODEL_DIR = "./model"
    TFLITE_OUTPUT_PATH = "./model.tflite"
    MAX_LENGTH = 128
    
    # Vérifier que le dossier du modèle existe
    if not os.path.exists(MODEL_DIR):
        print(f"❌ Erreur : Le dossier {MODEL_DIR} n'existe pas !")
        print("Assurez-vous que le dossier contient :")
        print("  - model.safetensors (ou pytorch_model.bin)")
        print("  - config.json") 
        print("  - tokenizer.json")
        print("  - tokenizer_config.json")
        exit(1)
    
    # Méthode principale
    print("🚀 Démarrage de la conversion SafeTensors -> TFLite")
    print("=" * 60)
    
    success = convert_safetensors_to_tflite(
        model_dir=MODEL_DIR,
        tflite_output_path=TFLITE_OUTPUT_PATH,
        max_length=MAX_LENGTH
    )
    
    # Si la méthode principale échoue, essayer la méthode alternative
    if not success:
        print("\n" + "=" * 60)
        print("🔄 Tentative avec méthode alternative...")
        success = convert_with_alternative_method(
            MODEL_DIR, 
            TFLITE_OUTPUT_PATH, 
            MAX_LENGTH
        )
    
    if success:
        print("\n🎉 Processus de conversion terminé avec succès !")
    else:
        print("\n💥 Toutes les méthodes de conversion ont échoué.")
        print("Suggestions :")
        print("1. Vérifiez que votre modèle est compatible")
        print("2. Essayez avec une version plus récente de TensorFlow")
        print("3. Utilisez un opset ONNX plus ancien (11 ou 12)")