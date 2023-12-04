# Run the project

# Mlrsnet
REPO_URL = https://github.com/DaryLu0v0/MLRSNet.git
CLONE_PATH = path/to/your/directory

mlrsnet_Download:
	@echo "Cloning MLRSNet repository..."
	@mkdir -p $(CLONE_PATH)
	@cd $(CLONE_PATH) && git clone $(REPO_URL)
	@echo "Unraring files..."
	@cd $(CLONE_PATH)/MLRSNet/images && for file in *.rar; do unrar e "$$file"; done

mrlsnet_Train_baseline:
	@echo "Running baseline training..."
	@python src/m_train_baseline.py > results/base_out.txt

mrlsnet_Train_with_different_data_size:
	@echo "Training with different data sizes..."
	@python src/m_test_different_size.py > results/data_size_out.txt
