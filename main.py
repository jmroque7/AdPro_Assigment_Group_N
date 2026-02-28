from app.okavango import OkavangoProject


def main() -> None:
    proj = OkavangoProject(root_dir=".", do_downloads=True)
    print("Datasets:", list(proj.dataframes.keys()))
    for name, gdf in proj.maps.items():
        print(name, "rows:", len(gdf), "missing %:", round(gdf["value"].isna().mean() * 100, 2))


if __name__ == "__main__":
    main()


