from src.gan.config import *

from src.gan.gan import GAN


def main():
    params = process_params()
    prepare_environment(params)
    gan = GAN(params)
    train_start = datetime.now()
    gan.train(load_data(params["data_path"], params["image_size"], params["image_channels"], params["batch_size"],
                        params["norm"], params["shuffle_data"], params["center_crop"], params["dl_workers"]))
    # Save models
    # Save generator net
    torch.save(gan.generator_net.state_dict(),
               f"{params['experiment_folder']}/models/{gan.generator_net.__class__.__name__}.pth")
    # Save discriminator net
    torch.save(gan.generator_net.state_dict(),
               f"{params['experiment_folder']}/models/{gan.discriminator_net.__class__.__name__}.pth")
    # Save loss plot
    save_loss_plot(gan.gen_loss, gan.disc_loss, f"{params['experiment_folder']}/plot")
    # Generate images
    generate_fake_images(generator=gan.generator_net, batch_size=params["batch_size"], device=gan.device,
                         noise_v_size=params["vector_size"],
                         images_save_path=f"{params['experiment_folder']}/generated_images")
    # Generate report
    create_report(batch_size=params["batch_size"], epochs=params["num_epochs"], gen_net=gan.generator_net,
                  disc_net=gan.discriminator_net, image_size=params["image_size"], optimizer=params["optimizer"],
                  dataset_name=params["data_path"].split()[-1], disc_loss=gan.disc_loss, gen_loss=gan.gen_loss,
                  loss_fn=gan.loss_fn, lr=params["learning_rate"], r_path=f"{params['experiment_folder']}/report",
                  s_date=train_start)


if __name__ == "__main__":
    main()
