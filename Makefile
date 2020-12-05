.PHONY: help
help: ## Show this help
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

test:  ## unit tests
	$(VAE_PY) -m pytest -vvv tests/

download_data:
	wget https://hosted-matrices-prod.s3-us-west-2.amazonaws.com/Single_cell_atlas_of_peripheral_immune_response_to_SARS_CoV_2_infection-25/Single_cell_atlas_of_peripheral_immune_response_to_SARS_CoV_2_infection.h5ad \
		-o data/raw/Single_cell_atlas_of_peripheral_immune_response_to_SARS_CoV_2_infection.h5ad

sparsify_data: ## convert Single_cell_atlas...h5ad to sparse matrix
	$(VAE_PY) src/pipeline/sparsify_covid_data.py
